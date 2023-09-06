import os
import yaml
from collections.abc import Iterable
from helper.util import OrderedYaml

Loader, Dumper = OrderedYaml()


def parse_option(args):
    with open(args.opt, mode='r') as f:
        opt = yaml.load(f, Loader=Loader)

    # export CUDA_VISIBLE_DEVICES
    if isinstance(opt['gpu'], Iterable):
        gpu_list = ','.join(str(x) for x in opt['gpu'])
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
        print('export CUDA_VISIBLE_DEVICES=' + gpu_list)

    opt['trial'] = args.trial

    exp_par = 'S_{}_T_{}_{}_{}_ctr{}cls{}p{}'.format(opt['student_model'],
                                                     opt['teacher_model'],
                                                     opt['dataset'],
                                                     opt['distill'],
                                                     opt['loss_ctr'],
                                                     opt['loss_cls'],
                                                     opt['loss_pred'])

    exp_root = os.path.join('./experiments', exp_par, 'trial_{}'.format(opt['trial']))
    opt['path']['exp_root'] = exp_root
    opt['path']['log'] = exp_root
    opt['path']['model'] = os.path.join(exp_root, 'model')
    opt['path']['state'] = os.path.join(exp_root, 'state')

    return opt


def save(opt):
    dump_dir = opt['path']['exp_root']
    dump_path = os.path.join(dump_dir, 'options.yml')
    with open(dump_path, 'w') as dump_file:
        yaml.dump(opt, dump_file, Dumper=Dumper)


def dict2str(opt, indent_l=1):
    '''dict to string for logger'''
    msg = ''
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_l * 2) + k + ':[\n'
            msg += dict2str(v, indent_l + 1)
            msg += ' ' * (indent_l * 2) + ']\n'
        else:
            msg += ' ' * (indent_l * 2) + k + ': ' + str(v) + '\n'
    return msg


class NoneDict(dict):
    def __missing__(self, key):
        return None


# convert to NoneDict, which return None for missing key.
def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt
