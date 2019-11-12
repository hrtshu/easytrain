
from tempfile import mktemp, mkdtemp

from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard


def tqdm_dummy(iterable, *args, **kwargs):
    yield from iterable


def fit(model, train_data, test_data=None, *, patience=None, max_epochs=1,
        monitor=None, use_weights='best', tb_log=None, callbacks=[],
        shuffle=True, verbose=0):
    if monitor is None:
        monitor = 'val_loss' if test_data else 'loss'
    if use_weights not in ('best', 'last'):
        raise ValueError("use_weights must be 'best' or 'last'")

    # TODO 事前にcallbacksに既にEarlyStoppingなどが含まれている場合に警告を出す
    callbacks = list(callbacks)
    if patience is not None:
        callbacks.append(EarlyStopping(patience=patience, monitor=monitor,
                                       verbose=int(bool(verbose))))
    if use_weights == 'best':
        best_model_path = mktemp()
        callbacks.append(ModelCheckpoint(best_model_path, save_best_only=True,
                                         save_weights_only=True,
                                         verbose=int(bool(verbose))))
    if tb_log:
        if type(tb_log) != str:
            tb_log = mkdtemp()
        callbacks.append(TensorBoard(log_dir=tb_log, histogram_freq=1))

    res_ = model.fit(*train_data, shuffle=shuffle, epochs=max_epochs,
                     callbacks=callbacks, validation_data=test_data,
                     verbose=int(verbose))
    history = res_.history

    if use_weights == 'best':
        model.load_weights(best_model_path)

    res = dict(model=model, history=history)

    if tb_log:
        res['tb_log_dir'] = tb_log

    return res


def cross_fit(model_builder, x, y,
              *, cv, fit_params={}, tqdm=None, verbose=0):
    if not tqdm:
        tqdm = tqdm_dummy

    for train_idx, test_idx in tqdm(cv.split(x), total=cv.get_n_splits(x)):
        train_x, train_y = x[train_idx], y[train_idx]
        test_x, test_y = x[test_idx], y[test_idx]

        # TODO 各分割の始めにverboseを表示する

        model = model_builder()

        res = fit(model, (train_x, train_y), (test_x, test_y),
                  **fit_params, verbose=verbose)
        res['train_idx'] = train_idx
        res['test_idx'] = test_idx

        yield res
