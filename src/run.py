import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd

from .model import TextClassifier
from torch.utils.data import TensorDataset, DataLoader
from sklearn.utils import compute_class_weight
from sklearn.metrics import classification_report

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        #limit of counter
        self.patience = patience
        
        #minimum delta value
        self.min_delta = min_delta
        
        #counter
        self.counter = 0

        #minimum validation loss
        self.min_val_loss = np.inf

    def early_stop(self, val_loss):
        #continue if validation loss is smaller than current minimum value
        if val_loss < self.min_val_loss:
            self.min_val_loss = val_loss
            self.counter = 0

        #add counter if validation loss is larger than current minimum value plus minimum delta
        elif val_loss - self.min_val_loss >= self.min_delta:
            self.counter += 1
            
            #return true if limit is reached
            if self.counter >= self.patience:
                return True

        #return false if early stopping criteria is not fulfilled
        return False

class Run:
    @staticmethod
    def train(
        model: TextClassifier,
        device,
        train_data_loader: DataLoader,
        val_data_loader: DataLoader,
        class_weights,
        learning_rate: float,
        save_path: str,
        epochs: int = 100,
        patience: int = 3,
        min_delta: int = 0.001
    ):
        #determine which layers to apply the L2 regularization
        conv1d_layers = ['conv1d_lists']
        constrained_params = list(filter(lambda kv: kv[0] in conv1d_layers, model.named_parameters()))
        
        #these parameters won't apply the L2 regularization
        normal_params = list(filter(lambda kv: kv[0] not in conv1d_layers, model.named_parameters()))

        #define optimizer
        optimizer = optim.Adadelta([{'params': [param[1] for param in normal_params]}, {'params': [param[1] for param in constrained_params], 'weight_decay': 3}], lr=learning_rate, rho=0.95)
        
        #define loss function with class weights
        loss_fn = nn.CrossEntropyLoss(weight=class_weights).to(device)

        #best accuracy
        best_accuracy = 0

        #define early stopper
        early_stopper = EarlyStopper(patience=patience, min_delta=min_delta)

        for _ in range(epochs):
            #activate training flag on model
            model = model.train()

            correct_predictions = 0

            #iterate over training data
            for x, y in train_data_loader:
                #move tokens and labels to GPU
                x = x.to(device)
                y = y.to(device)

                #output of model with given tokens
                outputs = model(x)

                #get value of labels predicted by model
                _, y_preds = torch.max(outputs, dim=1)

                #calculate loss
                loss = loss_fn(outputs, y)

                #clear gradients
                optimizer.zero_grad()

                #backpropagation
                loss.backward()

                #update optimizer step
                optimizer.step()

                #save predictions
                correct_predictions += torch.sum(y_preds == y)
            
            #evaluate model state for current epoch
            val_accuracy, val_loss = Run.evaluation(model, val_data_loader, loss_fn, device)

            if(val_accuracy > best_accuracy):
                #update best accuracy
                best_accuracy = val_accuracy
                
                #save current model state
                torch.save(model.state_dict(), save_path)

            #return if early stopping criteria is already met
            if(early_stopper.early_stop(val_loss)):
                return best_accuracy

        return best_accuracy

    @staticmethod
    def evaluation(model: TextClassifier, val_data_loader: DataLoader, loss_fn: torch.optim, device):
        #set evaluation flag on model
        model = model.eval()
        
        #list of accuracy
        accuracies = list()

        #list of loss
        losses = list()

        with torch.no_grad():
            for x, y in val_data_loader:
                #move tokens and labels to device
                x = x.to(device)
                y = y.to(device)

                #output of model with given tokens
                outputs = model(x)

                loss = loss_fn(outputs, y)
                losses.append(loss.item())

                #get value of labels predicted by model
                y_preds = torch.argmax(outputs, dim=1).flatten()
                
                #compute accuracy
                accuracy = (y_preds == y).cpu().numpy().mean()

                #append accuracy value
                accuracies.append(accuracy)

        #count average of accuracies and losses
        return np.mean(accuracies), np.mean(losses)

    @staticmethod
    def get_predictions(model: TextClassifier, test_data_loader: DataLoader, device, batch_size: int = 64):
        #set evaluation flag on model
        model = model.eval()
        
        real_values = []
        predictions = []

        with torch.no_grad():
            for x, y in test_data_loader:
                #move tokens and labels to device
                x = x.to(device)
                y = y.to(device)

                #output of model with given tokens
                outputs = model(x)

                #get value of labels predicted by model
                y_preds = torch.argmax(outputs, dim=1).flatten()

                #add original labels to list
                real_values.extend(y)

                #add predicted labels to list
                predictions.extend(y_preds)
        
        #move values to cpu
        predictions = torch.stack(predictions).cpu()
        real_values = torch.stack(real_values).cpu()

        return real_values, predictions
    
    def create_classification_report(real_values: list, predictions: list, class_names: list):
        #create classification report
        report = classification_report(y_true=real_values, y_pred=predictions, target_names=class_names, output_dict=True)
        
        #convert classification report to pandas dataframe
        df = pd.DataFrame(report).transpose()

        return df


