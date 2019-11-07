
from tempfile import mktemp, mkdtemp

from sklearn.model_selection import KFold
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard


def tqdm_dummy(iterable, *args, **kwargs):
    yield from iterable


def fit(model, train_data, test_data, *, patience=None, max_epochs=1,
        use_weights='best', tb_log_dir=None, callbacks=[], shuffle=True,
        verbose=0):
    if use_weights not in ('best', 'last'):
        raise ValueError("use_weights must be 'best' or 'last'")

    # TODO 事前にcallbacksに既にEarlyStoppingなどが含まれている場合に警告を出す
    callbacks = list(callbacks)
    if patience is not None:
        callbacks.append(EarlyStopping(patience=patience,
                                       verbose=int(bool(verbose))))
    if use_weights == 'best':
        best_model_path = mktemp()
        callbacks.append(ModelCheckpoint(best_model_path, save_best_only=True,
                                         save_weights_only=True,
                                         verbose=int(bool(verbose))))
    if tb_log_dir is not None:
        callbacks.append(TensorBoard(log_dir=tb_log_dir, histogram_freq=1))

    res = model.fit(*train_data, shuffle=shuffle, epochs=max_epochs,
                    callbacks=callbacks,
                    validation_data=test_data, verbose=int(verbose))
    history = res.history

    if use_weights == 'best':
        model.load_weights(best_model_path)

    return history


def train_split(model_generator, x, y, *,
                fit_params={}, n_splits=None,
                include_tb_log=False, tqdm=None, verbose=0):
    if not tqdm:
        tqdm = tqdm_dummy

    kf = KFold(n_splits=n_splits, shuffle=True)

    for train_idx, test_idx in tqdm(kf.split(x), total=kf.get_n_splits(x)):
        train_x, train_y = x[train_idx], y[train_idx]
        test_x, test_y = x[test_idx], y[test_idx]

        model = model_generator()

        tb_log_dir = mkdtemp() if include_tb_log else None

        history = fit(model, (train_x, train_y), (test_x, test_y),
                      **fit_params,
                      tb_log_dir=tb_log_dir, verbose=verbose)

        res = dict(model=model, history=history,
                   train_idx=train_idx, test_idx=test_idx)
        if include_tb_log:
            res['tb_log_dir'] = tb_log_dir

        yield res
