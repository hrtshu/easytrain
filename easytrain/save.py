
import json
from os.path import join, exists
from os import makedirs, listdir
from shutil import move
from itertools import count

import numpy as np
from keras.models import load_model as _load_model

from .train import fit, cross_fit


__all__ = [
    'fit_and_save',
    'cross_fit_and_save',
    'load_fit_result',
    'load_cross_fit_result',
]


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
            raise FileExistsError('The directory contents are not empty: {}'
                                  .format(path))
    else:
        makedirs(path)

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


def load_fit_result(path, *, load_model=False, load_idx=False):
    result = {}

    # load model
    if load_model:
        model_path = join(path, 'model.h5')
        result['model'] = _load_model(model_path)

    # load history
    history_path = join(path, 'history.json')
    with open(history_path) as f:
        result['history'] = json.load(f)  # TODO clsオプションにデコーダを指定する

    # load tb_log
    # TODO

    # load train_idx
    if load_idx:
        train_idx_path = join(path, 'train_idx.npy')
        result['train_idx'] = np.load(train_idx_path)

    # load valid_idx
    if load_idx:
        valid_idx_path = join(path, 'valid_idx.npy')
        result['valid_idx'] = np.load(valid_idx_path)

    return result


def fit_and_save(*args, path, **kwargs):
    res = fit(*args, **kwargs)
    save_fit_result(res, path)


def cross_fit_and_save(*args, path, split_name_format='split{split:02d}',
                       **kwargs):
    if exists(path):
        if listdir(path):
            raise FileExistsError('The directory contents are not empty: {}'
                                  .format(path))
    else:
        makedirs(path)

    for split, res in enumerate(cross_fit(*args, **kwargs)):
        split_path = join(path, split_name_format.format(split=split))
        save_fit_result(res, path=split_path)


def load_cross_fit_result(path, *, split_name_format='split{split:02d}',
                          **kwargs):
    result = []
    for split in count():
        split_name = split_name_format.format(split=split)
        split_path = join(path, split_name)
        if not exists(split_path):
            break
        split_result = load_fit_result(split_path, **kwargs)
        result.append(split_result)
    return result
