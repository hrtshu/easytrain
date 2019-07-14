
import json
from datetime import datetime
import os
from os.path import join
from shutil import move

import numpy as np

from .train import train_split


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def start(iterable, name=None, out_dir=None, tqdm=None):
    if name is None:
        name = datetime.now().isoformat()
    if out_dir is None:
        out_dir = os.getcwd()

    path = join(out_dir, name)
    os.mkdir(path)

    for train_name, data, target, model, train_params in tqdm(iterable):
        train_path = join(path, train_name)
        os.mkdir(train_path)

        for i, (model, fit_history, tb_log_dir) \
                in enumerate(train_split(model, data, target,
                                         **train_params, tqdm=tqdm)):
            split_path = join(train_path, 'split{:03d}'.format(i))
            model.save(split_path + '.h5', overwrite=False)
            move(tb_log_dir, split_path + '_tblog')
            with open(split_path + '.json', 'x') as f:
                json.dump(fit_history, f, cls=MyEncoder, indent=4)
