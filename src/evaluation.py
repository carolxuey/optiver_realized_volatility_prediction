import path_utils
import training_utils
from visualize import visualize_predictions


def evaluate_predictions(df_train, predictions_column):

    """
    Visualize predictions of the given model for every individual stock and entire training set

    Parameters
    ----------
    df_train [pandas.DataFrame of shape (n_samples)]: DataFrame with stock_id and prediction columns
    predictions_column (str): Predictions of the model
    """

    model_name = predictions_column.split("_")[0]

    for fold in sorted(df_train['fold'].unique()):
        _, val_idx = df_train.loc[df_train['fold'] != fold].index, df_train.loc[df_train['fold'] == fold].index
        fold_score = training_utils.rmspe_metric(df_train.loc[val_idx, 'target'], df_train.loc[val_idx, predictions_column])
        print(f'Fold {fold} - RMSPE: {fold_score:.6}')

    oof_score = training_utils.rmspe_metric(df_train['target'], df_train[predictions_column])
    print(f'{"-" * 30}\nOOF RMSPE: {oof_score:.6}\n{"-" * 30}')

    visualize_predictions(
        y_true=df_train['target'],
        y_pred=df_train[predictions_column],
        title=f'{model_name.upper()} Model - Global Predictions (OOF Score: {oof_score:.6})',
        path=f'{path_utils.MODELS_PATH}/{predictions_column.split("_")[0]}/{model_name}_global_predictions.png'
    )

    for stock_id in df_train['stock_id'].unique():

        df_stock = df_train.loc[df_train['stock_id'] == stock_id, :]
        stock_oof_score = training_utils.rmspe_metric(df_stock['target'], df_stock[predictions_column])
        print(f'Stock {stock_id} - OOF RMSPE: {stock_oof_score:.6}')

        visualize_predictions(
            y_true=df_stock['target'],
            y_pred=df_stock[predictions_column],
            title=f'{model_name.upper()} Model - Stock {stock_id} Predictions (OOF Score: {stock_oof_score:.6})',
            path=f'{path_utils.MODELS_PATH}/{predictions_column.split("_")[0]}/{model_name}_stock{stock_id}_predictions.png'
        )
