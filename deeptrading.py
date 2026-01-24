import argparse
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from time import time, strftime, gmtime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from LstmNet import *
from train_model import *

# Changing to mps if available
device = torch.device("mps" if torch.mps.is_available() else "cpu")
print('Device :', device)

# Parsing arguments
parser = argparse.ArgumentParser(
        prog = "DeepTrading",
        description = "LSTM based model for stock prediction"
        )

parser.add_argument("--dataset_path", type = str, required = True, help = "path to the dataset file")
parser.add_argument("--model_path", type = str, default = '', help = "path to the model (if we want to load a model)")
parser.add_argument("--lr", type = float, default = 1e-3, help = "learning rate of the SGD")
parser.add_argument("--batch_size", type = int, default = 8, help = "batch size during the learning")
#parser.add_argument("--momentum", type = float, default = 0, help = "momentum of the SGD")
parser.add_argument("--save_dir", type = str, default = "", help = "where to save the model, the logs and the configuration")
parser.add_argument("--nepochs", type = int, default = 10, help = "number of epochs to make")
parser.add_argument("--num_layers", type = int, default = 3, help = "number of sublayers in the lstm")
parser.add_argument("--hidden_dim", type = int, default = 128, help = "hidden dimension")
#parser.add_argument("--label_smoothing", type = float, default = 0.1, help = "label smoothing to prevent overfitting")
#parser.add_argument("--dropout", type = float, default = 0.1, help = "dropout in the transformer")
args = parser.parse_args()


# Create the directory containing the model, the logs, etc.
dir_name = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
out_dir = os.path.join(args.save_dir, dir_name)
os.makedirs(out_dir)

path_model = os.path.join(out_dir, "model.pth")
path_config = os.path.join(out_dir, "config.json")
path_logs = os.path.join(out_dir, "logs.json")


# Hyper parameters
dataset_path = 'data/stocks.csv'
n_stocks = 26
batch_size = args.batch_size
hidden_dim = args.hidden_dim
num_layers = args.num_layers
nepochs = args.nepochs


print('Dataset path :', dataset_path)
print('n_stocks :', n_stocks)
print('batch_size :', batch_size)

# Import the dataset
df = pd.read_csv(dataset_path)


# Removing zero-columns
df.drop(columns = ['Capital Gains', 'Dividends', 'Stock Splits'], inplace = True)
for i in range(1, n_stocks):
	df.drop(columns = [f'Dividends.{i}', f'Stock Splits.{i}'], inplace = True)

# Converting into a numpy array
dataset = df.to_numpy()

# Normalizing data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_dataset = scaler.fit_transform(df)

# Separating the train and test sets
train_dataset, test_dataset = train_test_split(
    scaled_dataset,
    test_size=0.15,
    shuffle=False
)
print(scaled_dataset)

# Get training and test size
train_size, input_dim = train_dataset.shape
test_size, _ = test_dataset.shape

print('train_size :', train_size)
print('test_size :', test_size)
print('n_features :', input_dim)

# Making a dataloader
#train_loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = False)
#test_loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = False)


####################
# Training a model #
####################


# Importing a LstmNet
model = LSTMNet(input_dim,
	hidden_size = hidden_dim,
	num_layers = num_layers,
	out_size = input_dim)

# Load weights if specified
if args.model_path != '':
    model.load_state_dict(torch.load(args.model_path, weights_only=True))
model.to(device)

print(model)
print("Number of Parameters :", sum(p.numel() for p in model.parameters()))

# Loss 
criterion = torch.nn.MSELoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr = args.lr)

train_model(model,
	train_dataset,
	test_dataset,
	criterion,
	optimizer,
	nepochs,
	path_logs = path_logs,
	path_model = path_model,
	batch_size = batch_size,
	device = device
	)

torch.save(model.state_dict(), path_model)

