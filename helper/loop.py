import sys
import time
import torch
from helper.util import AverageMeter, accuracy


def train(train_loader, model, criterion_ce, criterion_pred, optimizer, summary_writer, epoch, opt):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses_total = AverageMeter()

    top1 = AverageMeter()
    top5 = AverageMeter()

    ctr_top1 = AverageMeter()
    ctr_top5 = AverageMeter()

    # switch to train mode
    model.train()
    model.t_k_encoder.eval()
    end = time.time()

    for idx, (images, labels) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if opt['gpu'] is not None:
            labels = labels.cuda(opt['gpu'], non_blocking=True)
            images[0] = images[0].cuda(opt['gpu'], non_blocking=True)
            images[1] = images[1].cuda(opt['gpu'], non_blocking=True)
            images[2] = images[2].cuda(opt['gpu'], non_blocking=True)

        # compute output
        ctr_logits, ctr_labels, s_q_pred, s_q_pred_2, s_k, s_k_2, s_output = \
            model(im_sq=images[0], im_sk=images[1], im_tk=images[2])

        loss_ctr = criterion_ce(ctr_logits, ctr_labels)
        loss_cls = criterion_ce(s_output, labels)
        loss_pred = (criterion_pred(s_q_pred, s_k) + criterion_pred(s_q_pred_2, s_k_2)).mean()

        loss_total = opt['loss_ctr'] * loss_ctr + opt['loss_cls'] * loss_cls + opt['loss_pred'] * loss_pred
        losses_total.update(loss_total.item(), images[0].size(0))

        acc1, acc5 = accuracy(s_output, labels, topk=(1, 5))
        ctr_acc1, ctr_acc5 = accuracy(ctr_logits, ctr_labels, topk=(1, 5))  # acc1/acc5 are (K+1)-way accuracy

        top1.update(acc1[0], images[0].size(0))
        top5.update(acc5[0], images[0].size(0))
        ctr_top1.update(ctr_acc1[0], images[0].size(0))
        ctr_top5.update(ctr_acc5[0], images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % opt['print_freq'] == 0:
            print('Epoch: [{0}] || [{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, idx, len(train_loader), batch_time=batch_time,
                                                                 data_time=data_time, loss=losses_total, top1=top1,
                                                                 top5=top5))
            sys.stdout.flush()

    summary_writer.add_scalar("loss_train/total", losses_total.avg, epoch)
    summary_writer.add_scalar("acc_train/top_1", top1.avg, epoch)
    summary_writer.add_scalar("acc_train/top_5", top5.avg, epoch)
    summary_writer.add_scalar("acc_cocord/top_1", ctr_top1.avg, epoch)
    summary_writer.add_scalar("acc_cocord/top_5", ctr_top5.avg, epoch)

    print('[=== Training * Acc@1 {top1.avg:.3f} || Acc@5 {top5.avg:.3f} ===]'.format(top1=top1, top5=top5))


def validate(val_loader, model, criterion, summary_writer, epoch, opt):
    """validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (inputs, target) in enumerate(val_loader):

            inputs = inputs.float()
            if torch.cuda.is_available():
                inputs = inputs.cuda(opt['gpu'], non_blocking=True)
                target = target.cuda(opt['gpu'], non_blocking=True)

            # compute output
            output = model.s_q_encoder(inputs)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt['print_freq'] == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(idx, len(val_loader), batch_time=batch_time,
                                                                     loss=losses, top1=top1, top5=top5))

        summary_writer.add_scalar("loss_val/loss", losses.avg, epoch)
        summary_writer.add_scalar("acc_val/top_1", top1.avg, epoch)
        summary_writer.add_scalar("acc_val/top_5", top5.avg, epoch)

        print('[=== Validation * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} ===]'.format(top1=top1, top5=top5))

    return top1.avg
