import numpy as np
import pandas as pd
import lightgbm as lgb

import path_utils
import training_utils
from visualize import visualize_feature_importance, visualize_predictions


class LightGBMTrainer:

    def __init__(self, model_name, model_path, predictors, model_parameters, fit_parameters):

        self.model_name = model_name
        self.model_path = model_path
        self.predictors = predictors
        self.model_parameters = model_parameters
        self.fit_parameters = fit_parameters

    def train_and_validate(self, df_train):

        print(f'\n{"-" * 35}\nRunning LightGBM Model for Training\n{"-" * 35}\n')

        feature_importance = pd.DataFrame(
            data=np.zeros(len(self.predictors)),
            index=self.predictors,
            columns=['Importance']
        )

        for fold in sorted(df_train['fold'].unique()):

            trn_idx, val_idx = df_train.loc[df_train['fold'] != fold].index, df_train.loc[df_train['fold'] == fold].index
            train_dataset = lgb.Dataset(df_train.loc[trn_idx, self.predictors], label=df_train.loc[trn_idx, 'target'])
            val_dataset = lgb.Dataset(df_train.loc[val_idx, self.predictors], label=df_train.loc[val_idx, 'target'])

            model = lgb.train(
                params=self.model_parameters,
                train_set=train_dataset,
                valid_sets=[train_dataset, val_dataset],
                num_boost_round=self.fit_parameters['boosting_rounds'],
                early_stopping_rounds=self.fit_parameters['early_stopping_rounds'],
                fobj=training_utils.rmspe_loss_lgb,
                feval=[training_utils.rmspe_eval_lgb],
                verbose_eval=self.fit_parameters['verbose_eval']
            )
            model.save_model(f'{path_utils.MODELS_PATH}/{self.model_name}_fold{fold}', num_iteration=None, start_iteration=0, importance_type='gain')

            df_train.loc[val_idx, f'{self.model_name}_predictions'] = model.predict(df_train.loc[val_idx, self.predictors])
            feature_importance['Importance'] += (model.feature_importance(importance_type='gain') / df_train['fold'].nunique())
            fold_score = training_utils.rmspe_metric(df_train.loc[val_idx, 'target'], df_train.loc[val_idx, f'{self.model_name}_predictions'])
            print(f'\nFold {fold} - RMSPE: {fold_score:.6}\n')

        oof_score = training_utils.rmspe_metric(df_train['target'], df_train[f'{self.model_name}_predictions'])
        print(f'{"-" * 50}\nLightGBM OOF RMSPE: {oof_score:.6}\n{"-" * 50}')

        visualize_feature_importance(
            feature_importance=feature_importance,
            title='LightGBM - Feature Importance (Gain)',
            path=f'{path_utils.MODELS_PATH}/{self.model_name}_feature_importance.png'
        )

    def inference(self, df_train):

        print(f'\n{"-" * 36}\nRunning LightGBM Model for Inference\n{"-" * 36}')

        for fold in sorted(df_train['fold'].unique()):

            trn_idx, val_idx = df_train.loc[df_train['fold'] != fold].index, df_train.loc[df_train['fold'] == fold].index

            model = lgb.Booster(model_file=f'{self.model_path}/{self.model_name}_fold{fold}')
            df_train.loc[val_idx, f'{self.model_name}_predictions'] = model.predict(df_train.loc[val_idx, self.predictors])

            fold_score = training_utils.rmspe_metric(df_train.loc[val_idx, 'target'], df_train.loc[val_idx, f'{self.model_name}_predictions'])
            print(f'Fold {fold} - RMSPE: {fold_score:.6}')

        oof_score = training_utils.rmspe_metric(df_train['target'], df_train[f'{self.model_name}_predictions'])
        print(f'{"-" * 30}\nOOF RMSPE: {oof_score:.6}\n{"-" * 30}')
        for stock_id in df_train['stock_id'].unique():
            df_stock = df_train.loc[df_train['stock_id'] == stock_id, :]
            stock_oof_score = training_utils.rmspe_metric(df_stock['target'], df_stock[f'{self.model_name}_predictions'])
            print(f'Stock {stock_id} - RMSPE: {stock_oof_score:.6}')

        df_train[f'{self.model_name}_predictions'].to_csv(f'{self.model_path}/{self.model_name}_predictions.csv', index=False)
        visualize_predictions(
            y_true=df_train['target'],
            y_pred=df_train[f'{self.model_name}_predictions'],
            path=f'{self.model_path}/{self.model_name}_predictions.png'
        )

