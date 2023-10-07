# import the required libraries
import torch
from torch.utils.data import Dataset
from torch import nn
from GI_Wrist import GI_Wrist
from wrist_cnn import ConvolutionalNet
from train_and_test import train, test


def main_wrist(config):
    # use cuda if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    num_of_measurements = config.num_of_measurements
    # generate the data ### use this line only if you want to generate the data
    # preprocessing_wrist.generate_data("Processed_Dataset/", num_of_measurements, config.shape)

    # create the dataset
    path_ending = str(config.num_of_measurements) + "_" + config.shape + ".csv"
    csv_path = "../Processed_Dataset/new_dataset_" + path_ending
    wrist_gi_dataset = GI_Wrist(csv_path)

    # split the data to train and test
    number_of_samples = len(wrist_gi_dataset)
    # define batch size
    batch_size = config.batch_size
    # define the lengths of the train and test datasets to numbers divisible by the batch size
    train_len = (int(number_of_samples * 0.8) // batch_size) * batch_size  # 80% of the data for training
    test_len = (int(number_of_samples * 0.2) // batch_size) * batch_size  # 20% of the data for testing

    # create a new dataset with the desired number of samples (train_len + test_len)
    subset_dataset = torch.utils.data.Subset(wrist_gi_dataset, range(train_len + test_len))

    # split the dataset into train and test sets
    train_set, test_set = torch.utils.data.random_split(subset_dataset, [train_len, test_len])

    ### from here on it is the same as in main.py in GI_MNIST ###

    # create a data loader for the train set
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    # create data loader for the test set
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # create the network
    model = ConvolutionalNet(num_of_measurements, batch_size).to(device)
    # choose a loss function
    criterion = nn.CrossEntropyLoss()
    # choose an optimizer and learning rate for the training
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=0.0003)
    # choose the number of epochs
    number_of_epochs = config.epoch

    # Train the model
    model = train(model, train_loader, criterion, optimizer, number_of_epochs, batch_size)
    # Evaluate the trained model
    test_acc = test(model, test_loader, criterion, batch_size)
    return test_acc
