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
parser.add_argument("--label_smoothing", type = float, default = 0.0, help = "label smoothing to prevent overfitting")
parser.add_argument("--classif_only", action = 'store_true', help = "1 if we want only to classify only, 0 o/w")
#parser.add_argument("--dropout", type = float, default = 0.1, help = "dropout in the transformer")
args = parser.parse_args()


# Create the directory containing the model, the logs, etc.
dir_name = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
out_dir = os.path.join(args.save_dir, dir_name)
os.makedirs(out_dir)

path_model = os.path.join(out_dir, "model.pth")
path_model_classif = os.path.join(out_dir, "model_classif.pth")
path_config = os.path.join(out_dir, "config.json")
path_logs = os.path.join(out_dir, "logs.json")

with open(path_config, 'w') as f:
    json.dump(vars(args), f)
    f.write('\n')

# Hyper parameters
dataset_path = args.dataset_path
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

# Computing labels
df_growths = df[['Growth'] + [f'Growth.{i}' for i in range(1, n_stocks)]]
growths = df_growths.to_numpy()
labels = np.argmax(growths, axis = 1)
print('Growth size :', growths.shape)
print('Labels :', labels, 'of size', labels.shape)

# Normalizing data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_dataset = scaler.fit_transform(df)


# Separating the train and test sets
train_dataset, test_dataset = train_test_split(
    scaled_dataset,
    test_size=0.15,
    shuffle=False
)
# Separating the train and test sets
train_labels, test_labels = train_test_split(
    labels,
    test_size=0.15,
    shuffle=False
)

# Get training and test size
train_size, input_dim = train_dataset.shape
test_size, _ = test_dataset.shape

print('train_size :', train_size)
print('test_size :', test_size)
print('n_features :', input_dim)


####################
# Training a model #
####################


# Importing a LstmNet
model = LSTMNet(input_dim,
	hidden_size = hidden_dim,
	num_layers = num_layers,
	out_size = input_dim)

# Adding a final layer for classification
model_classif = Classifier(model, n_stocks)

# Load weights if specified
if args.model_path != '':
    model.load_state_dict(torch.load(args.model_path, weights_only=True))
    model_classif.load_state_dict(torch.load(args.model_path.replace('model.pth', 'model_classif.pth'), weights_only=True))

# changing to appropriate device
model.to(device)
model_classif.to(device)

print(model)
print("Number of Parameters :", sum(p.numel() for p in model.parameters()))

# Loss 
criterion_classif = torch.nn.CrossEntropyLoss(label_smoothing = args.label_smoothing)

criterion = torch.nn.MSELoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr = args.lr)

optimizer_classif = optim.Adam(model_classif.parameters(), lr = args.lr)

if not args.classif_only:
	# Part 1 of training
	print('-- Training part 1 --')
	train_lstm(model,
	    train_dataset,
	    test_dataset,
	    criterion,
	    optimizer,
	    nepochs,
	    path_logs = path_logs,
	    path_model = path_model,
	    batch_size = batch_size,
	    device = device)

print('-- Training part 2 --')
train_model(model_classif,
	train_dataset,
	test_dataset,
	train_labels,
	test_labels,
	criterion_classif,
	optimizer_classif,
	nepochs,
	path_logs = path_logs,
	path_model = path_model_classif,
	batch_size = batch_size,
	device = device
	)

torch.save(model.state_dict(), path_model)

