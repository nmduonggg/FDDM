import os.path as osp
import yaml
from utils.utils import OrderedYaml, get_timestamp
Loader, Dumper = OrderedYaml()


def parse(opt_path, root):
    with open(opt_path, mode='r') as f:
        opt = yaml.load(f, Loader=Loader)

    # datasets
    for phase, dataset in opt['datasets'].items():

        phase = phase.split('_')[0]
        dataset['phase'] = phase

    # path
    for key, path in opt['path'].items():
        if path and key in opt['path'] and key != 'strict_load':
            opt['path'][key] = osp.expanduser(path)

    if root is None:
        job_dir = osp.join(opt['job_dir'], get_timestamp() + '-' + opt['name'])
    else:
        job_dir = osp.join(osp.join(osp.abspath(root), 'exp'))


    opt['path']['job_dir'] = job_dir
    opt['path']['models'] = osp.join(job_dir, 'models')
    opt['path']['training_state'] = osp.join(job_dir, 'training_state')
    opt['path']['log'] = job_dir

    return opt


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