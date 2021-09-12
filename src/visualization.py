import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

import path_utils
import preprocessing_utils
from cnn1d_model import CNN1DModel


def visualize_learning_curve(training_losses, validation_losses, title, path=None):

    """
    Visualize learning curves of the models

    Parameters
    ----------
    training_losses [array-like of shape (n_epochs)]: Array of training losses computed after every epoch
    validation_losses [array-like of shape (n_epochs)]: Array of validation losses computed after every epoch
    title (str): Title of the plot
    path (str or None): Path of the output file (if path is None, plot is displayed with selected backend)
    """

    fig, ax = plt.subplots(figsize=(32, 8), dpi=100)
    sns.lineplot(
        x=np.arange(1, len(training_losses) + 1),
        y=training_losses,
        ax=ax,
        label='train_loss'
    )
    sns.lineplot(
        x=np.arange(1, len(validation_losses) + 1),
        y=validation_losses,
        ax=ax,
        label='val_loss'
    )
    ax.set_xlabel('Epochs/Steps', size=15, labelpad=12.5)
    ax.set_ylabel('Loss', size=15, labelpad=12.5)
    ax.tick_params(axis='x', labelsize=12.5, pad=10)
    ax.tick_params(axis='y', labelsize=12.5, pad=10)
    ax.legend(prop={'size': 18})
    ax.set_title(title, size=20, pad=15)

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close(fig)


def visualize_feature_importance(feature_importance, title, path=None):

    """
    Visualize feature importance of the models

    Parameters
    ----------
    feature_importance [pandas.DataFrame of shape (n_features)]: DataFrame of features as index and importances as values
    title (str): Title of the plot
    path (str or None): Path of the output file (if path is None, plot is displayed with selected backend)
    """

    feature_importance.sort_values(by='Importance', inplace=True, ascending=False)

    fig, ax = plt.subplots(figsize=(24, len(feature_importance)))
    sns.barplot(
        x='Importance',
        y=feature_importance.index,
        data=feature_importance,
        palette='Blues_d',
        ax=ax
    )
    ax.set_xlabel('')
    ax.tick_params(axis='x', labelsize=12.5, pad=10)
    ax.tick_params(axis='y', labelsize=12.5, pad=10)
    ax.set_title(title, size=20, pad=15)

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close(fig)


def visualize_predictions(y_true, y_pred, title, path=None):

    """
    Visualize predictions of the models

    Parameters
    ----------
    y_true [array-like of shape (n_samples)]: Ground-truth
    y_pred [array-like of shape (n_samples)]: Predictions
    title (str): Title of the plot
    path (str or None): Path of the output file (if path is None, plot is displayed with selected backend)
    """

    fig, axes = plt.subplots(ncols=2, figsize=(32, 8))
    fig.subplots_adjust(top=0.85)
    sns.scatterplot(x=y_true, y=y_pred, ax=axes[0])
    sns.histplot(y_true, label='Ground-truth Labels', kde=True, color='blue', ax=axes[1])
    sns.histplot(y_pred, label='Predictions', kde=True, color='red', ax=axes[1])
    axes[0].set_xlabel(f'Ground-truth Labels', size=15, labelpad=12.5)
    axes[0].set_ylabel(f'Predictions', size=15, labelpad=12.5)
    axes[1].set_xlabel('')
    axes[1].set_ylabel('')
    axes[1].legend(prop={'size': 17.5})
    for i in range(2):
        axes[i].tick_params(axis='x', labelsize=12.5, pad=10)
        axes[i].tick_params(axis='y', labelsize=12.5, pad=10)
    axes[0].set_title(f'Ground-truth Labels vs Predictions', size=20, pad=15)
    axes[1].set_title(f'Predictions Distributions', size=20, pad=15)

    fig.suptitle(title, size=20, y=0.95)

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close(fig)


def visualize_time_bucket(df_train, stock_id, time_id, model_name, model_parameters, path):

    """
    Visualize time buckets as images

    Parameters
    ----------
    df_train [pandas.DataFrame of shape (n_samples, 2)]: Training DataFrame with stock_id and time_id columns
    stock_id (int): ID of the stock (0 <= stock_id <= 126)
    time_id (int): ID of the time bucket (0 <= stock_id <= 32767)
    model_name (str): Name of the model
    model_parameters (dict): Model configuration
    path (str or None): Path of the output file (if path is None, plot is displayed with selected backend)
    """

    sequences, current_realized_volatility, target = preprocessing_utils.read_time_bucket(
        df_train=df_train,
        stock_id=stock_id,
        time_id=time_id
    )

    sample = df_train.loc[(df_train['stock_id'] == stock_id) & (df_train['time_id'] == time_id), :]
    fold = int(sample['fold'].values[0])
    stock_id_encoded = int(sample['stock_id_encoded'].values[0])

    if model_name == 'cnn1d':
        model = CNN1DModel(**model_parameters)
    else:
        model = None

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.load_state_dict(torch.load(f'{path_utils.MODELS_PATH}/{model_name}/{model_name}_fold{fold}.pt'))
    model.to(device)
    model.eval()

    sequences_tensor = torch.as_tensor(sequences.reshape(1, 600, 25), dtype=torch.float)
    sequences_tensor = sequences_tensor.to(device)
    stock_id_encoded_tensor = torch.as_tensor([stock_id_encoded], dtype=torch.long)
    stock_id_encoded_tensor = stock_id_encoded_tensor.to(device)

    with torch.no_grad():
        prediction = model(stock_id_encoded_tensor, sequences_tensor).detach().cpu().numpy()[0]

    fig, ax = plt.subplots(figsize=(24, 60), dpi=200)
    ax.imshow(sequences)
    ax.set_xlabel('Sequences', size=15, labelpad=12.5)
    ax.set_ylabel('Timesteps', size=15, labelpad=12.5)
    ax.set_xticks([])
    ax.tick_params(axis='y', labelsize=12.5, pad=10)
    ax.set_title(
        f'Stock {stock_id} Time Bucket {time_id} - Current Realized Volatility: {current_realized_volatility:.6f} - Target: {target:.6f} - Prediction {prediction:.6f}',
        size=20,
        pad=15
    )

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close(fig)
