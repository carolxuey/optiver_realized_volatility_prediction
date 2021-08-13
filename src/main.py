import yaml
import argparse
import pandas as pd

import path_utils
import preprocessing_utils
from preprocessing import PreprocessingPipeline
from regular_trainer import RegularTrainer
from nested_trainer import NestedTrainer


if __name__ == '__main__':

    config = yaml.load(open('../rnn_config.yaml', 'r'), Loader=yaml.FullLoader)

    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str)
    args = parser.parse_args()

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
        **config['preprocessing']
    )
    df_train, df_test = preprocessing_pipeline.transform()

    print(f'\nProcessed Training Set Shape: {df_train.shape}')
    print(f'Processed Training Set Memory Usage: {df_train.memory_usage().sum() / 1024 ** 2:.2f} MB')
    print(f'Processed Test Set Shape: {df_test.shape}')
    print(f'Processed Test Set Memory Usage: {df_test.memory_usage().sum() / 1024 ** 2:.2f} MB')

    if config['model_type'] == 'regular':
        trainer = RegularTrainer(
            model_name=config['model_name'],
            model_path=config['model_path'],
            model_parameters=config['model'],
            training_parameters=config['training']
        )
    else:
        trainer = NestedTrainer(
            model_name=config['model_name'],
            model_path=config['model_path'],
            model_parameters=config['model'],
            training_parameters=config['training']
        )

    if args.mode == 'train':
        trainer.train_and_validate(df_train)
    elif args.mode == 'inference':
        trainer.inference(df_train, df_test)
