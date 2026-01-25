import torch
import time


def train_model(model,
    train_loader,
    test_loader,
    optimizer,
    criterion,
    num_epochs = 10,
    device = 'cuda' if torch.cuda.is_available() else 'cpu',
    batch_size = 64,
    ):
    train_losses = []
    test_losses = []
    train_acc = []
    test_acc = []
    time_spent = 0

    model.to(device)

    for epoch in range(num_epochs):
        time_start = time.time()
        model.train()
        train_loss = 0
        accuracy = 0

        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            
            # Forward pass - juste x maintenant, pas besoin de y comme input
            output = model(x)  # [batch, seq_len, num_classes]
            
            # y est de shape [batch, seq_len, num_classes] (one-hot ou probabilités)
            # Calculer la loss sur toute la séquence
            loss = criterion(output.reshape(-1, output.size(-1)), y.argmax(dim=-1).reshape(-1))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Accuracy
            preds = output.argmax(dim=-1)  # [batch, seq_len]
            targets = y.argmax(dim=-1)  # [batch, seq_len]
            correct = (preds == targets).float()
            accuracy += correct.mean().item()
            
            if (i+1) % 100 == 0:
                print(f'  Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}')
                
        train_loss /= len(train_loader)
        accuracy /= len(train_loader)
        train_losses.append(train_loss)
        train_acc.append(accuracy)
        time_end = time.time()
        time_spent += time_end - time_start
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {accuracy:.4f}, Time: {time_end - time_start:.2f}s')
        
        # Validation
        model.eval()
        test_loss = 0
        accuracy = 0
        with torch.no_grad():
            for i, (x, y) in enumerate(test_loader):
                x, y = x.to(device), y.to(device)
                
                output = model(x)
                loss = criterion(output.reshape(-1, output.size(-1)), y.argmax(dim=-1).reshape(-1))
                
                test_loss += loss.item()

                preds = output.argmax(dim=-1)
                targets = y.argmax(dim=-1)
                correct = (preds == targets).float()
                accuracy += correct.mean().item()
                
        test_loss /= len(test_loader)
        accuracy /= len(test_loader)
        test_losses.append(test_loss)
        test_acc.append(accuracy)
        print(f'Epoch {epoch+1}/{num_epochs}, Test Loss: {test_loss:.4f}, Test Acc: {accuracy:.4f}')
        
        if (epoch + 1) % 1 == 0:
            torch.save(model.state_dict(), f'model_temp_training.pt')
            
    print(f'Total Training Time: {time_spent:.2f}s')
    return train_losses, test_losses, train_acc, test_acc
