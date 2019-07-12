
from tempfile import mktemp

from sklearn.model_selection import KFold
from keras.callbacks import EarlyStopping, ModelCheckpoint


DEF_N_SPLITS = 10
DEF_PATIENCE = 10
DEF_MAX_EPOCHS = 1000


def tqdm_dummy(iterable, *args, **kwargs):
    yield from iterable


def train_split(model, x, y, n_splits=None, patience=None,
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

        best_model_path = mktemp()

        early_stopping = EarlyStopping(patience=patience, verbose=1)
        checkpoint = ModelCheckpoint(best_model_path, save_best_only=True,
                                     save_weights_only=True, verbose=1)

        res = model.fit(train_x, train_y, shuffle=True, epochs=max_epochs,
                        callbacks=[early_stopping, checkpoint],
                        validation_data=(test_x, test_y))
        history = res.history

        model.load_weights(best_model_path)

        yield model, history
