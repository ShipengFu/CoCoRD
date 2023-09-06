import os
import random
from collections import OrderedDict
from collections.abc import Iterable
from datetime import datetime
import shutil
import numpy as np
import torch
import torch.nn as nn
import yaml
from models import model_dict
import time
import torch.nn.functional as F
try:
    from yaml import CDumper as Dumper
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Dumper, Loader


def OrderedYaml():
    """yaml orderedDict support"""
    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def get_network_description(network):
    """Get the string and total parameters of the network"""
    if isinstance(network, nn.DataParallel):
        network = network.module
    s = str(network)
    n = sum(map(lambda x: x.numel(), network.parameters()))

    return s, n


def print_network(model, exp_root):
    """
    print network summary including module and number of parameters
    """
    s, n = get_network_description(model)
    if isinstance(model, nn.DataParallel):
        net_struc_str = '{} - {}'.format(model.__class__.__name__,
                                         model.module.__class__.__name__)
    else:
        net_struc_str = '{}'.format(model.__class__.__name__)

    print("==================================================")
    print("===> Network Summary\n")
    net_lines = []
    line = s + '\n'
    print(line)
    net_lines.append(line)
    line = 'Network structure: [{}], with parameters: [{:,d}]'.format(net_struc_str, n)
    print(line)
    net_lines.append(line)

    with open(os.path.join(exp_root, 'network_summary.txt'), 'w') as f:
        f.writelines(net_lines)

    print("==================================================")


def print_network_test(model):
    """
    print network summary including module and number of parameters
    """
    s, n = get_network_description(model)
    if isinstance(model, nn.DataParallel):
        net_struc_str = '{} - {}'.format(model.__class__.__name__,
                                         model.module.__class__.__name__)
    else:
        net_struc_str = '{}'.format(model.__class__.__name__)

    print("==================================================")
    print("===> Network Summary\n")
    net_lines = []
    line = s + '\n'
    print(line)
    net_lines.append(line)
    line = 'Network structure: [{}], with parameters: [{:,d}]'.format(net_struc_str, n)
    print(line)
    net_lines.append(line)
    print("==================================================")


def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(opt['lr_decay_epochs']))
    if steps > 0:
        new_lr = opt['learning_rate'] * (opt['lr_decay_rate'] ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, Iterable):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir_and_rename(path):
    if os.path.exists(path):
        new_name = path + '_archived_' + get_timestamp()
        print('Path already exists. Rename it to [{:s}]'.format(new_name))
        os.rename(path, new_name)
    os.makedirs(path)


def load_t_model(model_path, n_cls, opt):
    print('===> Loading teacher model...')
    model = model_dict[opt['teacher_model']](num_classes=n_cls)

    model.load_state_dict(torch.load(model_path, map_location='cuda:{}'.format(opt['gpu']))['model'])
    for param in model.parameters():
        param.requires_grad = False
    print('===> Loading & freezing teacher model done...')
    return model


def validate_teacher(val_loader, model, opt):
    """validation"""
    batch_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (inputs, target) in enumerate(val_loader):

            inputs = inputs.float()
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                target = target.cuda()

            # compute output
            # output = model.module.t_k_encoder(inputs)
            output = model.t_k_encoder(inputs)
            # output = model(inputs)
            # measure accuracy
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt['print_freq'] == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time,
                       top1=top1, top5=top5))

    return top1.avg, top5.avg


def save_checkpoint(state, is_best, opt):
    filename = os.path.join(opt['path']['model'], 'checkpoint.pth')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(opt['path']['model'], 'best.pth'))


class DistillMSE(nn.Module):

    def __init__(self):
        super(DistillMSE, self).__init__()

    def forward(self, x, y):
        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)
        return 2 - 2 * (x * y).sum(dim=-1)
