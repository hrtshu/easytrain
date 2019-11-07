
from tempfile import mktemp, mkdtemp

from sklearn.model_selection import KFold
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard


DEF_N_SPLITS = 10
DEF_PATIENCE = 10
DEF_MAX_EPOCHS = 1000


def tqdm_dummy(iterable, *args, **kwargs):
    yield from iterable


def fit(model, train_data, test_data, *, patience, max_epochs,
        use_weights='best', tb_log_dir=None, callbacks=[], verbose=0):
    if use_weights not in ('best', 'last'):
        raise ValueError("use_weights must be 'best' or 'last'")

    # TODO 事前にcallbacksに既にEarlyStoppingなどが含まれている場合に警告を出す
    callbacks = list(callbacks)
    callbacks.append(EarlyStopping(patience=patience,
                                   verbose=int(bool(verbose))))
    if use_weights == 'best':
        best_model_path = mktemp()
        callbacks.append(ModelCheckpoint(best_model_path, save_best_only=True,
                                         save_weights_only=True,
                                         verbose=int(bool(verbose))))
    if tb_log_dir is not None:
        callbacks.append(TensorBoard(log_dir=tb_log_dir, histogram_freq=1))

    res = model.fit(*train_data, shuffle=True, epochs=max_epochs,
                    callbacks=callbacks,
                    validation_data=test_data, verbose=int(verbose))
    history = res.history

    if use_weights == 'best':
        model.load_weights(best_model_path)

    return history


def train_split(model_generator, x, y, n_splits=None, patience=None,
                max_epochs=None, tqdm=None):
    if not n_splits:
        n_splits = DEF_N_SPLITS
    if patience is None:
        patience = DEF_PATIENCE
    if not max_epochs:
        max_epochs = DEF_MAX_EPOCHS
    if not tqdm:
        tqdm = tqdm_dummy

    kf = KFold(n_splits=n_splits, shuffle=True)

    for train_idx, test_idx in tqdm(kf.split(x), total=kf.get_n_splits(x)):
        train_x, train_y = x[train_idx], y[train_idx]
        test_x, test_y = x[test_idx], y[test_idx]

        model = model_generator()

        tb_log_dir = mkdtemp()

        history = fit(model, (train_x, train_y), (test_x, test_y),
                      patience=patience, max_epochs=max_epochs,
                      use_weights='best', tb_log_dir=tb_log_dir, verbose=1)

        yield model, history, tb_log_dir
