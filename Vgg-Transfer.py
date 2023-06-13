# PyTorch
from torchvision import transforms, datasets, models
import torch
from torch.nn import (
    Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax,
    BatchNorm2d, Dropout
)
from torch import optim, cuda
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from sklearn.model_selection import train_test_split
import glob
from PIL import Image
from sklearn import preprocessing
from torchvision.models import VGG16_Weights
from torch.optim import Adam
from timeit import default_timer as timer

# Data science tools
import numpy as np
import pandas as pd

# Useful for examining network
from torchsummary import summary

# Visualizations
import matplotlib.pyplot as plt


    
def train_model(model,criterion,optimizer,train_loader,valid_loader,save_file_name,
          max_epochs_stop=3,n_epochs=20,print_every=1):
    """Train a PyTorch Model

    Params
    --------
        model (PyTorch model): cnn to train
        criterion (PyTorch loss): objective to minimize
        optimizer (PyTorch optimizier): optimizer to compute gradients of model parameters
        train_loader (PyTorch dataloader): training dataloader to iterate through
        valid_loader (PyTorch dataloader): validation dataloader used for early stopping
        save_file_name (str ending in '.pt'): file path to save the model state dict
        max_epochs_stop (int): maximum number of epochs with no improvement in validation loss for early stopping
        n_epochs (int): maximum number of training epochs
        print_every (int): frequency of epochs to print training stats

    Returns
    --------
        model (PyTorch model): trained cnn with best weights
        history (DataFrame): history of train and validation loss and accuracy
    """

    # Early stopping intialization
    epochs_no_improve = 0
    valid_loss_min = np.Inf

    valid_max_acc = 0
    history = []

    # Number of epochs already trained (if using loaded in model weights)
    try:
        print(f'Model has been trained for: {model.epochs} epochs.\n')
    except:
        model.epochs = 0
        print(f'Starting Training from Scratch.\n')

    overall_start = timer()

    # Main loop
    for epoch in range(n_epochs):

        # keep track of training and validation loss each epoch
        train_loss = 0.0
        valid_loss = 0.0

        train_acc = 0
        valid_acc = 0

        # Set to training
        model.train()
        start = timer()

        # Training loop
        for ii, (data, target) in enumerate(train_loader):


            # Clear gradients
            optimizer.zero_grad()
            # Predicted outputs are log probabilities
            output = model(data)

            # Loss and backpropagation of gradients
            loss = criterion(output, target)
            loss.backward()

            # Update the parameters
            optimizer.step()

            # Track train loss by multiplying average loss by number of examples in batch
            train_loss += loss.item() * data.size(0)

            # Calculate accuracy by finding max log probability
            _, pred = torch.max(output, dim=1)
            correct_tensor = pred.eq(target.data.view_as(pred))
            # Need to convert correct tensor from int to float to average
            accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
            # Multiply average accuracy times the number of examples in batch
            train_acc += accuracy.item() * data.size(0)

            # Track training progress
            print(
                f'Epoch: {epoch}\t{100 * (ii + 1) / len(train_loader):.2f}% complete. {timer() - start:.2f} seconds elapsed in epoch.',
                end='\r')

        # After training loops ends, start validation
        else:
            model.epochs += 1

            # Don't need to keep track of gradients
            with torch.no_grad():
                # Set to evaluation mode
                model.eval()

                # Validation loop
                for data, target in valid_loader:

                    # Forward pass
                    output = model(data)

                    # Validation loss
                    loss = criterion(output, target)
                    # Multiply average loss times the number of examples in batch
                    valid_loss += loss.item() * data.size(0)

                    # Calculate validation accuracy
                    _, pred = torch.max(output, dim=1)
                    correct_tensor = pred.eq(target.data.view_as(pred))
                    accuracy = torch.mean(
                        correct_tensor.type(torch.FloatTensor))
                    # Multiply average accuracy times the number of examples
                    valid_acc += accuracy.item() * data.size(0)

                # Calculate average losses
                train_loss = train_loss / len(train_loader.dataset)
                valid_loss = valid_loss / len(valid_loader.dataset)

                # Calculate average accuracy
                train_acc = train_acc / len(train_loader.dataset)
                valid_acc = valid_acc / len(valid_loader.dataset)

                history.append([train_loss, valid_loss, train_acc, valid_acc])

                # Print training and validation results
                if (epoch + 1) % print_every == 0:
                    print(
                        f'\nEpoch: {epoch} \tTraining Loss: {train_loss:.4f} \tValidation Loss: {valid_loss:.4f}'
                    )
                    print(
                        f'\t\tTraining Accuracy: {100 * train_acc:.2f}%\t Validation Accuracy: {100 * valid_acc:.2f}%'
                    )

                # Save the model if validation loss decreases
                if valid_loss < valid_loss_min:
                    # Save model
                    torch.save(model, save_file_name)
                    # Track improvement
                    epochs_no_improve = 0
                    valid_loss_min = valid_loss
                    valid_best_acc = valid_acc
                    best_epoch = epoch

                # Otherwise increment count of epochs with no improvement
                else:
                    epochs_no_improve += 1
                    # Trigger early stopping
                    if epochs_no_improve >= max_epochs_stop:
                        print(
                            f'\nEarly Stopping! Total epochs: {epoch}. Best epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%'
                        )
                        total_time = timer() - overall_start
                        print(
                            f'{total_time:.2f} total seconds elapsed. {total_time / (epoch+1):.2f} seconds per epoch.'
                        )

                        # Load the best state dict
                        model.load_state_dict(torch.load(save_file_name))
                        # Attach the optimizer
                        model.optimizer = optimizer

                        # Format history
                        history = pd.DataFrame(
                            history,
                            columns=[
                                'train_loss', 'valid_loss', 'train_acc',
                                'valid_acc'
                            ])
                        return model, history

    # Attach the optimizer
    model.optimizer = optimizer
    # Record overall time and print out stats
    total_time = timer() - overall_start
    print(
        f'\nBest epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%'
    )
    print(
        f'{total_time:.2f} total seconds elapsed. {total_time / (epoch):.2f} seconds per epoch.'
    )
    # Format history
    history = pd.DataFrame(
        history,
        columns=['train_loss', 'valid_loss', 'train_acc', 'valid_acc'])
    return model, history

            
def gather_png():

    def find_matching_string(input_string, string_list):
        for string in string_list:
            if string in input_string:
                return string
        return None  

    preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    dir_path = "rgb"
    # csv files in the path
    files = glob.glob(dir_path + "/*.png")
    
    # defining an empty list to store 
    # content
    content = []
    letters=['Gamma', 'Beta', 'Eta', 'Phi', 'Theta', 'Xi', 'Zeta']
    le = preprocessing.LabelEncoder()
    le.fit(letters)
    label_list = []
    
    # checking all the csv files in the 
    # specified path
    for filename in files:
        matching_string = find_matching_string(filename, letters)
            
        # reading content of csv file
        with open(filename, 'rb') as image:
            data = Image.open(image)
            input_tensor = preprocess(data)

            content.append(input_tensor)
            if matching_string:
                label_list.append(matching_string)
        
    label_list=le.fit_transform(label_list)
    content=torch.stack(content)
    print('Dataset size:', content.size)
    print('Lables size:', len(label_list))

    return content, np.array(label_list, dtype=np.int32)

def get_pretrained_model(model_name, n_classes):

    if model_name == 'vgg16':
        model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

        # Freeze early layers
        for param in model.parameters():
            param.requires_grad = False
        n_inputs = model.classifier[6].in_features

        # Add on classifier
        model.classifier[6] = nn.Sequential(
            nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, n_classes), nn.LogSoftmax(dim=1))

    return model

def main():
    #Defining the parameters
    save_file_name = 'vgg16-transfer.pth'
    checkpoint_path = 'vgg16-transfer.pth'

    # Change to fit hardware
    #!CHECK
    batch_size = 50

    data_array, label_list=gather_png()
    print('Array: \n:', data_array)
    print('Shape: \n', len(data_array))
    print(label_list)
    
    train_x, val_x, train_y, val_y= train_test_split(data_array, label_list, test_size = 0.1)
    train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.2)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train = TensorDataset(torch.tensor(train_x), torch.tensor(train_y).long())
    test = TensorDataset(torch.tensor(test_x), torch.tensor(test_y).long())
    val = TensorDataset(torch.tensor(val_x), torch.tensor(val_y).long())
    
    # Dataloader iterators
    dataloaders = {
        'train': DataLoader(train, batch_size=batch_size, shuffle=True),
        'val': DataLoader(val, batch_size=batch_size, shuffle=True),
        'test': DataLoader(test, batch_size=batch_size, shuffle=True)
    }

    trainiter = iter(dataloaders['train'])
    features, labels = next(trainiter)
    print(features)
    print(features.shape)
    print(labels)
    print(labels.shape)

    model = get_pretrained_model('vgg16', n_classes=7)
    print(summary(model, input_size=(3, 224, 224), batch_size=batch_size, device='cpu'))
    
    #Until here it works
    optimizer = Adam(model.parameters(), lr=0.01)
    criterion = CrossEntropyLoss()

    model, history = train_model(
        model,
        criterion,
        optimizer,
        dataloaders['train'],
        dataloaders['val'],
        save_file_name=save_file_name,
        max_epochs_stop=5,
        n_epochs=5,
        print_every=1)


if __name__ == "__main__":
    main()