import argparse
import random
import warnings

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import helper.util as util
import cocord.builder
import options.options as option
from dataset.cifar100 import get_cifar100_dataloaders
from helper.loop import train, validate
import time


def main():

    parser = argparse.ArgumentParser('arguments for CoCoRD training')
    parser.add_argument('-opt', type=str, required=True, help='path to option yaml file')
    parser.add_argument('-trial', '-t', type=int, default=0, help='trial for experiment id')
    args = parser.parse_args()
    opt = option.parse_option(args)

    util.mkdir_and_rename(opt['path']['exp_root'])  # rename experiment folder if exists
    util.mkdirs((path for key, path in opt['path'].items() if not key == 'exp_root'
                                                               and 'pretrain_model' not in key
                                                               and 'resume' not in key 
                                                               and not key == 'teacher'))
    option.save(opt)

    if opt['seed'] is not None:
        random.seed(opt['seed'])
        torch.manual_seed(opt['seed'])
        torch.cuda.manual_seed_all(opt['seed'])
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    best_acc = 0.
    best_epoch = 0

    print("Use GPU: {} for training".format(opt['gpu']))
    summary_writer = SummaryWriter(log_dir=opt['path']['log'])

    # =================== model construction ==========
    cocord_kd = cocord.builder.CoCoRD(opt=opt)  # === build the entire cocord model ===
    util.print_network(cocord_kd, opt['path']['log'])

    torch.cuda.set_device(opt['gpu'])
    cocord_kd = cocord_kd.cuda(opt['gpu'])
    
    # =================== criterion & optimizer ===================
    criterion_ce = nn.CrossEntropyLoss().cuda(opt['gpu'])
    criterion_pred = util.DistillMSE().cuda(opt['gpu'])

    optimizer = optim.SGD(cocord_kd.parameters(), lr=opt['learning_rate'],
                          momentum=opt['momentum'], weight_decay=opt['weight_decay'])
    
    cudnn.benchmark = True

    # =================== data loader ===================
    if opt['dataset'].lower() == 'cifar100':
        train_loader, val_loader = get_cifar100_dataloaders(opt)
    else:
        raise NotImplementedError(opt['dataset'])
    
    t_acc1, _ = util.validate_teacher(val_loader, cocord_kd, opt)
    print(f'Teacher acc: [{t_acc1:.2f}]')

    # =================== training routine ===================
    for epoch in range(opt['start_epoch']+1, opt['epochs']+1):

        util.adjust_learning_rate(epoch, opt, optimizer)
        print(f"current learning rate: {optimizer.param_groups[0]['lr']:.5f}")

        time_start = time.time()
        train(train_loader, cocord_kd, criterion_ce, criterion_pred, optimizer, summary_writer, epoch, opt)
        val_acc = validate(val_loader, cocord_kd, criterion_ce, summary_writer, epoch, opt)
        print('Elapsed time: [{elapsed_time:.2f}(s)]'.format(elapsed_time=time.time()-time_start))

        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        if is_best:
            best_epoch = epoch

        util.save_checkpoint({
                'epoch': epoch,
                's_model_dict': cocord_kd.s_q_encoder.state_dict(),
                'state_dict': cocord_kd.state_dict(),
                'best_acc1': best_acc,
            }, is_best, opt)
        
        print(f"Best_epoch: [{best_epoch}] || Best_acc: [{best_acc:.2f}]\n")

    print('===> Training Finished...')


if __name__ == '__main__':
    main()
