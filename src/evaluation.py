import training_utils


def evaluate_predictions(df_train, predictions_column):

    for fold in sorted(df_train['fold'].unique()):
        _, val_idx = df_train.loc[df_train['fold'] != fold].index, df_train.loc[df_train['fold'] == fold].index
        fold_score = training_utils.rmspe_metric(df_train.loc[val_idx, 'target'], df_train.loc[val_idx, predictions_column])
        print(f'Fold {fold} - RMSPE: {fold_score:.6}')

    oof_score = rmspe_metric(df_train['target'], df_train[predictions_column])
    print(f'{"-" * 30}\nOOF RMSPE: {oof_score:.6}\n{"-" * 30}')

    for stock_id in df_train['stock_id'].unique():
        df_stock = df_train.loc[df_train['stock_id'] == stock_id, :]
        stock_oof_score = rmspe_metric(df_stock['target'], df_stock[predictions_column])
        print(f'Stock {stock_id} - OOF RMSPE: {stock_oof_score:.6}')