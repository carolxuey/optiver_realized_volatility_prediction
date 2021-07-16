import numpy as np
from sklearn.model_selection import StratifiedKFold, GroupKFold


def get_stratified_folds(df_train, n_splits, shuffle=True, random_state=42, verbose=False):

    """
    Create a column of fold numbers with specified configuration of stratified k-fold on given training set

    Parameters
    ----------
    df_train [pandas.DataFrame of shape (428932, 3)]: Training set
    n_splits (int): Number of folds (2 <= n_splits)
    shuffle (bool): Whether to shuffle data before split or not
    random_state (int): Random seed for reproducibility
    verbose (bool): Flag for verbosity
    """

    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    for fold, (_, val_idx) in enumerate(skf.split(X=df_train, y=df_train['stock_id']), 1):
        df_train.loc[val_idx, 'stratified_fold'] = fold
    df_train['stratified_fold'] = df_train['stratified_fold'].astype(np.uint8)

    if verbose:
        print(f'Training set split into {n_splits} stratified folds')
        for fold in range(1, n_splits + 1):
            df_fold = df_train[df_train['stratified_fold'] == fold]
            print(f'Fold {fold} {df_fold.shape} - target mean: {df_fold["target"].mean():.4} std: {df_fold["target"].std():.4} min: {df_fold["target"].min():.4} max: {df_fold["target"].max():.4}')


def get_group_folds(df_train, n_splits, verbose=False):

    """
    Create a column of fold numbers with specified configuration of group k-fold on given training set

    Parameters
    ----------
    df_train [pandas.DataFrame of shape (428932, 3)]: Training set
    n_splits (int): Number of folds (2 <= n_splits)
    verbose (bool): Flag for verbosity
    """

    gkf = GroupKFold(n_splits=n_splits)
    for fold, (_, val_idx) in enumerate(gkf.split(X=df_train, groups=df_train['time_id']), 1):
        df_train.loc[val_idx, 'group_fold'] = fold
    df_train[f'group_fold'] = df_train['group_fold'].astype(np.uint8)

    if verbose:
        print(f'Training set split into {n_splits} group folds')
        for fold in range(1, n_splits + 1):
            df_fold = df_train[df_train['group_fold'] == fold]
            print(f'Fold {fold} {df_fold.shape} - target mean: {df_fold["target"].mean():.4} std: {df_fold["target"].std():.4} min: {df_fold["target"].min():.4} max: {df_fold["target"].max():.4}')
