import torch
import helper.util as util
import torch.nn as nn
from models import model_dict


def make_projector(in_dim, hidden_dim, out_dim):
    projector = nn.Sequential(
        nn.Linear(in_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, out_dim)
    )

    return projector


def make_predictor(dim, pred_dim):
    predictor = nn.Sequential(
        nn.Linear(dim, pred_dim), nn.BatchNorm1d(pred_dim), nn.ReLU(inplace=True),
        nn.Linear(pred_dim, dim))
    return predictor


class CoCoRD(nn.Module):

    def __init__(self, opt):
        super(CoCoRD, self).__init__()

        self.K = opt['cocord']['key_size']
        self.m = opt['cocord']['momentum']
        self.T = opt['cocord']['T']
        self.b_m = opt['cocord']['b_momentum']

        # ======== create the encoders ========
        self.s_q_encoder = model_dict[opt['student_model']](num_classes=opt['n_cls'])
        self.s_k_encoder = model_dict[opt['student_model']](num_classes=opt['n_cls'])
        self.t_k_encoder = util.load_t_model(opt['path']['teacher'], opt['n_cls'], opt)  # === build teacher model ===

        # ======== build projectors ========
        if opt['student_model'] in ['ShuffleV1', 'ShuffleV2']:
            s_feat_dim = self.s_q_encoder.linear.in_features
        elif opt['student_model'] in ['vgg8', 'MobileNext', 'MobileNetV2']:
            s_feat_dim = self.s_q_encoder.classifier.in_features
        else:
            s_feat_dim = self.s_q_encoder.fc.in_features

        if opt['teacher_model'] in ['ResNet50']:
            t_feat_dim = self.t_k_encoder.linear.in_features
        elif opt['teacher_model'] in ['vgg13']:
            t_feat_dim = self.t_k_encoder.classifier.in_features
        else:
            t_feat_dim = self.t_k_encoder.fc.in_features

        self.s_q_proj = make_projector(in_dim=s_feat_dim, hidden_dim=opt['cocord']['hidden_dim'],
                                       out_dim=opt['cocord']['embedding_dim'])
        self.s_k_proj = make_projector(in_dim=s_feat_dim, hidden_dim=opt['cocord']['hidden_dim'],
                                       out_dim=opt['cocord']['embedding_dim'])
        self.t_k_proj = make_projector(in_dim=t_feat_dim, hidden_dim=opt['cocord']['hidden_dim'],
                                       out_dim=opt['cocord']['embedding_dim'])

        # ======== build predictors ========
        self.s_q_pred = make_predictor(dim=opt['cocord']['embedding_dim'], pred_dim=opt['cocord']['pred_dim'])

        # ======== copy param. from query encoder to key encoder ========
        for param_sq, param_sk in zip(self.s_q_encoder.parameters(), self.s_k_encoder.parameters()):
            param_sk.data.copy_(param_sq.data)  # initialize
            param_sk.requires_grad = False  # not update by gradient

        for param_sq_p, param_sk_p in zip(self.s_q_proj.parameters(), self.s_k_proj.parameters()):
            param_sk_p.data.copy_(param_sq_p.data)  # initialize
            param_sk_p.requires_grad = False  # not update by gradient

        for param_sq_p, param_tk_p in zip(self.s_q_proj.parameters(), self.t_k_proj.parameters()):
            param_tk_p.data.copy_(param_sq_p.data)  # initialize
            param_tk_p.requires_grad = False  # not update by gradient

        # ======== create the queue ========
        self.register_buffer("queue", torch.randn(opt['cocord']['embedding_dim'], opt['cocord']['key_size']))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update
        """
        for param_sq_en, param_sk_en in zip(self.s_q_encoder.parameters(), self.s_k_encoder.parameters()):
            param_sk_en.data = param_sk_en.data * self.b_m + param_sq_en.data * (1. - self.b_m)

        for param_sq_proj, param_sk_proj in zip(self.s_q_proj.parameters(), self.s_k_proj.parameters()):
            param_sk_proj.data = param_sk_proj.data * self.b_m + param_sq_proj.data * (1. - self.b_m)

        for param_sq_proj, param_tk_proj in zip(self.s_q_proj.parameters(), self.t_k_proj.parameters()):
            param_tk_proj.data = param_tk_proj.data * self.m + param_sq_proj.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):

        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        if (ptr + batch_size) <= self.K:
            self.queue[:, ptr:ptr + batch_size] = keys.T  # === replace the keys at ptr (dequeue and enqueue) ===
            ptr = (ptr + batch_size) % self.K  # === move pointer ===
            self.queue_ptr[0] = ptr
        else:
            move_size = self.K - ptr
            self.queue[:, ptr:] = keys[:move_size, :].T
            ptr = (ptr + batch_size) % self.K
            self.queue[:, :ptr] = keys[move_size:, :].T
            self.queue_ptr[0] = ptr

    def forward(self, im_sq, im_sk, im_tk):

        # ======== compute s_query & t_key features ========
        s_q, s_output = self.s_q_encoder(im_sq, is_logits=True)
        s_q = self.s_q_proj(s_q)  # === queries: N x embedding_dim ===
        s_q_norm = nn.functional.normalize(s_q, dim=1)

        # ======== compute targets & predictions ========
        s_q_2, _ = self.s_q_encoder(im_sk, is_logits=True)
        s_q_2 = self.s_q_proj(s_q_2)  # === compute predictions from im_t_k

        s_q_pred = self.s_q_pred(s_q)  # === std_pred: N x pred_dim
        s_q_pred_2 = self.s_q_pred(s_q_2)  # === std_pred_2: N x pred_dim

        # ======== compute s_key features ========
        with torch.no_grad():  # === no gradient to keys ===
            self._momentum_update_key_encoder()  # === update the encoder ===

            t_k, _ = self.t_k_encoder(im_tk, is_logits=True)
            t_k = self.t_k_proj(t_k)  # === keys: N x embedding_dim ===
            t_k = nn.functional.normalize(t_k, dim=1)

            s_k, _ = self.s_k_encoder(im_sk, is_logits=True)
            s_k_2, _ = self.s_k_encoder(im_sq, is_logits=True)

            s_k = self.s_k_proj(s_k)  # === compute targets for s_q_pred
            s_k_2 = self.s_k_proj(s_k_2)  # === compute targets for s_q_pred_2

        # ======== compute logits ========
        # Einstein sum is more intuitive
        l_pos = torch.einsum('nc,nc->n', [s_q_norm, t_k]).unsqueeze(-1)  # === positive logits: Nx1
        l_neg = torch.einsum('nc,ck->nk', [s_q_norm, self.queue.clone().detach()])  # === negative logits: NxK ===

        logits = torch.cat([l_pos, l_neg], dim=1)  # === logits: Nx(1+K) ===
        logits /= self.T  # === apply temperature ===
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()  # === labels: positive key indicators ===

        self._dequeue_and_enqueue(t_k)  # === dequeue and enqueue ===

        return logits, labels, s_q_pred, s_q_pred_2, s_k.detach(), s_k_2.detach(), s_output
