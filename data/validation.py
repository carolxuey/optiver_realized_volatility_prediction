import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold


def get_stratified_folds(df, n_splits, shuffle=True, random_state=42, verbose=False):

    """
    Create a column of fold numbers with specified configuration of stratified k-fold on given training set

    Parameters
    ----------
    df [pandas.DataFrame of shape (n_samples, 3)]: Training set
    n_splits (int): Number of folds (2 <= n_splits)
    shuffle (bool): Whether to shuffle data before split or not
    random_state (int): Random seed for reproducibility
    verbose (bool): Flag for verbosity
    """

    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    for fold, (_, val_idx) in enumerate(skf.split(X=df, y=df['stock_id']), 1):
        df.loc[val_idx, 'fold_stratified'] = fold
    df['fold_stratified'] = df['fold_stratified'].astype(np.uint8)

    if verbose:
        print(f'\nTraining set split into {n_splits} stratified folds')
        for fold in range(1, n_splits + 1):
            df_fold = df[df['fold_stratified'] == fold]
            print(f'Fold {fold} {df_fold.shape} - target mean: {df_fold["target"].mean():.4} std: {df_fold["target"].std():.4} min: {df_fold["target"].min():.4} max: {df_fold["target"].max():.4}')


def get_group_folds(df, n_splits, verbose=False):

    """
    Create a column of fold numbers with specified configuration of group k-fold on given training set

    Parameters
    ----------
    df [pandas.DataFrame of shape (n_samples, 3)]: Training set
    n_splits (int): Number of folds (2 <= n_splits)
    verbose (bool): Flag for verbosity
    """

    gkf = GroupKFold(n_splits=n_splits)
    for fold, (_, val_idx) in enumerate(gkf.split(X=df, groups=df['time_id']), 1):
        df.loc[val_idx, 'fold_group'] = fold
    df['fold_group'] = df['fold_group'].astype(np.uint8)

    if verbose:
        print(f'\nTraining set split into {n_splits} group folds')
        for fold in range(1, n_splits + 1):
            df_fold = df[df['fold_group'] == fold]
            print(f'Fold {fold} {df_fold.shape} - target mean: {df_fold["target"].mean():.4} std: {df_fold["target"].std():.4} min: {df_fold["target"].min():.4} max: {df_fold["target"].max():.4}')


def get_random_folds(df, n_splits, shuffle=True, random_state=42, verbose=False):

    """
    Create a column of fold numbers with specified configuration of k-fold on given partial training set

    Parameters
    ----------
    df [pandas.DataFrame of shape (n_samples, 3)]: Training set
    n_splits (int): Number of folds (2 <= n_splits)
    shuffle (bool): Whether to shuffle data before split or not
    random_state (int): Random seed for reproducibility
    verbose (bool): Flag for verbosity
    """

    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    for stock_id in sorted(df['stock_id'].unique()):
        df_stock = df.loc[df['stock_id'] == stock_id, :].reset_index(drop=True)
        for fold, (_, val_idx) in enumerate(kf.split(X=df_stock), 1):
            df_stock.loc[val_idx, 'fold_nested'] = fold
        df.loc[df['stock_id'] == stock_id, 'fold_nested'] = df_stock['fold_nested'].values
    df['fold_nested'] = df['fold_nested'].astype(np.uint8)

    if verbose:
        print(f'\nTraining set split into {n_splits} random folds inside every stock')
        for fold in range(1, n_splits + 1):
            df_fold = df[df['fold_nested'] == fold]
            print(f'Fold {fold} {df_fold.shape} - target mean: {df_fold["target"].mean():.4} std: {df_fold["target"].std():.4} min: {df_fold["target"].min():.4} max: {df_fold["target"].max():.4}')


if __name__ == '__main__':

    df_train = pd.read_csv('./train.csv')

    get_stratified_folds(df_train, 5, verbose=True)
    get_group_folds(df_train, 5, verbose=True)
    get_random_folds(df_train, 5, verbose=True)
    df_train[['fold_stratified', 'fold_group', 'fold_nested']].to_csv('folds.csv', index=False)
