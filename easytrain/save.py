
import json
from os.path import join, exists
from os import mkdir, listdir
from shutil import move

import numpy as np

from .train import fit, cross_fit


__all__ = ['fit_and_save', 'cross_fit_and_save']


class _JSONEncoderForNumpy(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(_JSONEncoderForNumpy, self).default(obj)


def save_fit_result(result, *, path):
    if exists(path):
        if listdir(path):
            raise FileExistsError('The directory contents are not empty.')
    else:
        mkdir(path)

    # save model
    model_path = join(path, 'model.h5')
    result['model'].save(model_path, overwrite=False)

    # save history
    history_path = join(path, 'history.json')
    with open(history_path, 'x') as f:
        json.dump(result['history'], f, cls=_JSONEncoderForNumpy, indent=4)

    # save tb_log
    if 'tb_log_dir' in result:
        tb_log_path = join(path, 'tb_log')
        move(result['tb_log_dir'], tb_log_path)

    # save train_idx
    if 'train_idx' in result:
        train_idx_path = join(path, 'train_idx.npy')
        np.save(train_idx_path, result['train_idx'])

    # save valid_idx
    if 'valid_idx' in result:
        valid_idx_path = join(path, 'valid_idx.npy')
        np.save(valid_idx_path, result['valid_idx'])


def fit_and_save(*args, path, **kwargs):
    res = fit(*args, **kwargs)
    save_fit_result(res, path)


def cross_fit_and_save(*args, path, split_name_format='split{split:02d}',
                       **kwargs):
    if exists(path):
        if listdir(path):
            raise FileExistsError('The directory contents are not empty.')
    else:
        mkdir(path)

    for split, res in enumerate(cross_fit(*args, **kwargs)):
        split_path = join(path, split_name_format.format(split=split))
        save_fit_result(res, path=split_path)
