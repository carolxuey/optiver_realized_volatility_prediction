import yaml
import argparse
import pandas as pd

import path_utils
import preprocessing_utils
from preprocessing import PreprocessingPipeline
from nn_trainer import NeuralNetworkTrainer
from lgb_trainer import LightGBMTrainer
import evaluation


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('mode', type=str)
    args = parser.parse_args()

    config = yaml.load(open(f'../{args.model}_config.yaml', 'r'), Loader=yaml.FullLoader)

    df_train = pd.read_csv(
        f'{path_utils.DATA_PATH}/train.csv',
        dtype=preprocessing_utils.train_test_dtypes['train']
    )
    df_test = pd.read_csv(
        f'{path_utils.DATA_PATH}/test.csv',
        usecols=['stock_id', 'time_id'],
        nrows=1,
        dtype=preprocessing_utils.train_test_dtypes['test']
    )

    preprocessing_pipeline = PreprocessingPipeline(
        df_train,
        df_test,
        config['preprocessing']['create_features'],
        config['preprocessing']['split_type'],
        config['preprocessing']['n_splits'],
        config['preprocessing']['shuffle'],
        config['preprocessing']['random_state']
    )
    df_train, df_test = preprocessing_pipeline.transform()

    print(f'\nProcessed Training Set Shape: {df_train.shape}')
    print(f'Processed Training Set Memory Usage: {df_train.memory_usage().sum() / 1024 ** 2:.2f} MB')
    print(f'Processed Test Set Shape: {df_test.shape}')
    print(f'Processed Test Set Memory Usage: {df_test.memory_usage().sum() / 1024 ** 2:.2f} MB')

    trainer = None

    if args.model == 'cnn1d' or args.model == 'rnn':

        trainer = NeuralNetworkTrainer(
            model_name=config['model_name'],
            model_path=config['model_path'],
            preprocessing_parameters=config['preprocessing'],
            model_parameters=config['model'],
            training_parameters=config['training']
        )

    elif args.model == 'lgb':

        trainer = LightGBMTrainer(
            model_name=config['model_name'],
            model_path=config['model_path'],
            predictors=config['predictors'],
            model_parameters=config['model'],
            fit_parameters=config['fit']
        )

    if args.mode == 'train':
        trainer.train_and_validate(df_train)
    elif args.mode == 'inference':
        trainer.inference(df_train)
        evaluation.evaluate_predictions(df_train, predictions_column=f'{config["model_name"]}_predictions')
