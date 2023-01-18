import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd

from .model import TextClassifier
from torch.utils.data import TensorDataset, DataLoader
from sklearn.utils import compute_class_weight
from sklearn.metrics import classification_report

class Run:
    @staticmethod
    def train(
        model: TextClassifier,
        device,
        data: dict,
        learning_rate: float,
        save_path: str,
        epochs: int = 10,
        batch_size: int = 30
    ):
        #create dataset instances for training and validation
        train = TensorDataset(torch.tensor(data['x_train']), torch.tensor(data['y_train']))
        val = TensorDataset(torch.tensor(data['x_val']), torch.tensor(data['y_val']))

        #create data loader for training and validation
        train_data_loader = DataLoader(train, batch_size=batch_size)
        val_data_loader = DataLoader(val, batch_size=batch_size)

        #define optimizer
        optimizer = optim.Adadelta(model.parameters(), lr=learning_rate, rho=0.95)
        
        #compute class weights
        weights = compute_class_weight(class_weight='balanced', classes=np.unique(data['y_train']), y=data['y_train'])
        class_weights = torch.FloatTensor(weights).cuda()

        #define loss function with class weights
        loss_fn = nn.CrossEntropyLoss(weight=class_weights).to(device)

        #best accuracy
        best_accuracy = 0

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
            val_accuracy = Run.evaluation(model, val_data_loader, device)

            if(val_accuracy > best_accuracy):
                #update best accuracy
                best_accuracy = val_accuracy
                
                #save current model state
                torch.save(model.state_dict(), save_path)

        return best_accuracy

    @staticmethod
    def evaluation(model: TextClassifier, val_data_loader: DataLoader, device):
        #set evaluation flag on model
        model = model.eval()
        
        #list of accuracy
        accuracies = list()

        with torch.no_grad():
            for x, y in val_data_loader:
                #move tokens and labels to device
                x = x.to(device)
                y = y.to(device)

                #output of model with given tokens
                outputs = model(x)

                #get value of labels predicted by model
                y_preds = torch.argmax(outputs, dim=1).flatten()
                
                #compute accuracy
                accuracy = (y_preds == y).cpu().numpy().mean()

                #append accuracy value
                accuracies.append(accuracy)

        #count average of accuracies
        return np.mean(accuracies)

    @staticmethod
    def get_predictions(model: TextClassifier, data, device, batch_size: int = 64):
        #create test dataset instance
        test = TensorDataset(torch.tensor(data['x_test']), torch.tensor(data['y_test']))
        
        #create test data loader
        test_data_loader = DataLoader(test, batch_size=batch_size)
        
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


