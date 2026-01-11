import torch
import torch.nn as nn
from time import time
from numpy import inf
import json

# Function to train the model
def train_model(model,
    train_dataset,
    test_dataset,
    criterion,
    optimizer,
    nepochs,
    modulo = 1,
    path_logs = None,
    path_model = None,
    batch_size = 8,
    device = None):
    
    if device is None:
        device = 'cpu'
    # Save loss and training time
    train_losses = []
    test_losses = []
    total_time = 0

    for epoch in range(nepochs):
        time_start = time()
        train_loss = 0.
        test_loss = 0.

        ###################
        # Train the model #
        ###################
        model.train()

        # Useful to compute accuracy
        dataset_size = len(train_dataset)

        for k in range(0, dataset_size, batch_size):
            # Put the data on the appropriate device
            if k == 0:
                hn = torch.zeros(model.num_layers, model.hidden_size)
                cn = torch.zeros(model.num_layers, model.hidden_size)
                hn = hn.to(device = device)
                cn = cn.to(device = device)
                continue
            inp = torch.from_numpy(train_dataset[k-batch_size:k]).to(torch.float32)
            label = torch.from_numpy(train_dataset[k - batch_size + 1 : k + 1]).to(torch.float32)

            inp = inp.to(device = device)
            label = label.to(device = device)
            
            # Computation of the output
            output, (hn, cn) = model(inp, (hn, cn))

            # Computation of the loss
            loss = criterion(output, label) * batch_size / dataset_size
            train_loss += loss.item()

            # Making an optimizer step every batch_size steps
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            hn = hn.detach()
            cn = cn.detach()

        ##################
        # Test the model #
        ##################
        model.eval()

        # Useful to compute accuracy
        dataset_size = len(test_dataset)

        for k in range(0, dataset_size, batch_size):
            with torch.no_grad():
                # Put the data on the appropriate device
                if k == 0:
                    hn = torch.zeros(model.num_layers, model.hidden_size)
                    cn = torch.zeros(model.num_layers, model.hidden_size)
                    hn = hn.to(device = device)
                    cn = cn.to(device = device)
                    continue
                inp = torch.from_numpy(test_dataset[k-batch_size:k]).to(torch.float32)
                label = torch.from_numpy(test_dataset[k - batch_size + 1 : k + 1]).to(torch.float32)

                inp = inp.to(device = device)
                label = label.to(device = device)
                
                # Computation of the output
                output, (hn, cn) = model(inp, (hn, cn))
    
                # Computation of the loss
                loss = criterion(output, label) * batch_size / dataset_size
                test_loss += loss.item()
    

        ###################
        # Storing results #
        ###################
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        total_time += time() - time_start

        # Saving the logs
        d = {
            "Epoch" : epoch,
            "Training Loss" : train_loss,
            "Validation Loss" : test_loss,
            #"Training Acc" : train_accuracy,
            #"Validation Acc" : test_accuracy,
            "Time" : time() - time_start,
            "Total time" : total_time
        }

        if path_logs is not None:
            with open(path_logs, "a") as f:
                json.dump(d, f)
                f.write('\n')

        # Saving the model
        if path_model is not None:
            torch.save(model.state_dict(), path_model)

        # Log the results
        if epoch % modulo == 0:
            print('Epoch: {} \tTraining Loss: {:.6f} \tTest Loss: {:.6f} '.format(epoch, train_loss, test_loss))
            # print('Epoch: {} \tTraining Loss: {:.6f} \tTest Loss: {:.6f} \tTraining acc: {:.6f} \tTest acc: {:.6f}'.format(epoch, train_loss, test_loss, train_accuracy, test_accuracy))
    return train_losses, test_losses #, train_accuracies, test_accuracies, total_time






