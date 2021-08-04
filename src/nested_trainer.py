import os
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch.optim as optim

import path_utils
import training_utils
from datasets import Optiver2DNestedDataset
from cnn1d_model import CNN1DNestedModel
from visualize import draw_learning_curve


class NestedTrainer:

    def __init__(self, model_name, model_path, model_parameters, training_parameters):

        self.model_name = model_name
        self.model_path = model_path
        self.model_parameters = model_parameters
        self.training_parameters = training_parameters

    def get_model(self):

        model = None

        if self.model_name == 'cnn1d':
            model = CNN1DNestedModel(**self.model_parameters)

        return model

    def train_fn(self, train_loader, model, criterion, optimizer, device):

        print('\n')
        model.train()
        progress_bar = tqdm(train_loader)
        losses = []

        if self.training_parameters['amp']:
            scaler = torch.cuda.amp.GradScaler()
        else:
            scaler = None

        for sequences, target in progress_bar:
            sequences, target = sequences.to(device), target.to(device)

            if scaler is not None:
                with torch.cuda.amp.autocast():
                    optimizer.zero_grad()
                    output = model(sequences)
                    loss = criterion(target, output)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.zero_grad()
                output = model(sequences)
                loss = criterion(target, output)
                loss.backward()
                optimizer.step()

            losses.append(loss.item())
            average_loss = np.mean(losses)
            progress_bar.set_description(f'train_rmspe: {average_loss:.6f}')

        train_loss = np.mean(losses)
        return train_loss

    def val_fn(self, val_loader, model, criterion, device):

        model.eval()
        progress_bar = tqdm(val_loader)
        losses = []

        with torch.no_grad():
            for sequences, target in progress_bar:
                sequences, target = sequences.to(device), target.to(device)
                output = model(sequences)
                loss = criterion(target, output)
                losses.append(loss.item())
                average_loss = np.mean(losses)
                progress_bar.set_description(f'val_rmspe: {average_loss:.6f}')

        val_loss = np.mean(losses)
        return val_loss

    def train_and_validate(self, df_train, train_stock=None):

        print(f'\n{"-" * 26}\nRunning Model for Training\n{"-" * 26}\n')

        model_root_dir = self.model_path
        if not os.path.exists(model_root_dir):
            os.makedirs(model_root_dir)

        for stock_id in sorted(df_train['stock_id'].unique()):

            if train_stock is not None:
                if train_stock != stock_id:
                    continue

            model_stock_dir = os.path.join(model_root_dir, f'stock_{stock_id}')
            if not os.path.exists(model_stock_dir):
                os.makedirs(model_stock_dir)

            for fold in sorted(df_train['fold'].unique()):

                print(f'\nFold {fold} - Stock {stock_id}\n{"-" * 16}')

                trn_idx, val_idx = df_train.loc[df_train['fold'] != fold].index, df_train.loc[df_train['fold'] == fold].index
                train_dataset = Optiver2DNestedDataset(df=df_train.loc[trn_idx, :], stock_id=stock_id)
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=self.training_parameters['batch_size'],
                    sampler=RandomSampler(train_dataset),
                    pin_memory=True,
                    drop_last=False,
                    num_workers=self.training_parameters['num_workers'],
                )
                val_dataset = Optiver2DNestedDataset(df=df_train.loc[val_idx, :], stock_id=stock_id)
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=self.training_parameters['batch_size'],
                    sampler=SequentialSampler(val_dataset),
                    pin_memory=True,
                    drop_last=False,
                    num_workers=self.training_parameters['num_workers'],
                )

                training_utils.set_seed(self.training_parameters['random_state'], deterministic_cudnn=self.training_parameters['deterministic_cudnn'])
                device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
                model = self.get_model()
                model = model.to(device)

                criterion = training_utils.rmspe_loss_pt
                optimizer = optim.Adam(
                    model.parameters(),
                    lr=self.training_parameters['learning_rate'],
                    betas=self.training_parameters['betas'],
                    weight_decay=self.training_parameters['weight_decay']
                )
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                    patience=self.training_parameters['reduce_lr_patience'],
                    factor=self.training_parameters['reduce_lr_factor'],
                    min_lr=self.training_parameters['reduce_lr_min'],
                    verbose=True
                )

                early_stopping = False
                summary = {
                    'train_loss': [],
                    'val_loss': []
                }

                for epoch in range(1, self.training_parameters['epochs'] + 1):

                    if early_stopping:
                        break

                    train_loss = self.train_fn(train_loader, model, criterion, optimizer, device)
                    val_loss = self.val_fn(val_loader, model, criterion, device)
                    print(f'Epoch {epoch} - Training Loss: {train_loss:.6f} - Validation Loss: {val_loss:.6f}')
                    scheduler.step(val_loss)

                    best_val_loss = np.min(summary['val_loss']) if len(summary['val_loss']) > 0 else np.inf
                    if val_loss < best_val_loss:
                        model_path = f'{model_stock_dir}/{self.model_name}_fold{fold}.pt'
                        torch.save(model.state_dict(), model_path)
                        print(f'Saving model to {model_path} (validation loss decreased from {best_val_loss:.6f} to {val_loss:.6f})')

                    summary['train_loss'].append(train_loss)
                    summary['val_loss'].append(val_loss)

                    best_iteration = np.argmin(summary['val_loss']) + 1
                    if len(summary['val_loss']) - best_iteration >= self.training_parameters['early_stopping_patience']:
                        print(f'Early stopping (validation loss didn\'t increase for {self.training_parameters["early_stopping_patience"]} epochs/steps)')
                        print(f'Best validation loss is {np.min(summary["val_loss"]):.6f}')
                        draw_learning_curve(
                            training_losses=summary['train_loss'],
                            validation_losses=summary['val_loss'],
                            title=f'{self.model_name} - Stock {stock_id} Fold {fold} Learning Curve',
                            path=f'{model_stock_dir}/{self.model_name}_fold{fold}_learning_curve.png'
                        )
                        early_stopping = True
