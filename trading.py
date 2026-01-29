import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

def trading_test(data_init,data,model_classifier,num_layers, hidden_dim,value_init = None,value_data = None): # data of size (n_jours, n_indices)
    '''
    Test des stratégies de trading, chaque jour on a un budget fixé, et on considère qu'on achète des fractions d'indices, on compare le modèle avec des stratégies baseline.
    '''
    n_jours, _= data.shape
    _, n_indices = value_data.shape
    # stratégie naîve : on achète le plus gros indice chaque jour
    l_indices_achats_naif = np.zeros(n_indices)
    l_argent_naif = []
    for i in range(n_jours-1):
        #best_index = np.argmax(value_data[i])
        best_index = 1
        l_indices_achats_naif[best_index] += 10/value_data[i][best_index]
        if value_data is None:
            l_argent_naif.append(np.sum(l_indices_achats_naif*data[i+1]))
        else:
            l_argent_naif.append(np.sum(l_indices_achats_naif*value_data[i+1]))
    #Stratégie améliorée : acheter la plus grosse croissance
    l_indices_achats_reg = np.zeros(n_indices)
    l_argent_reg = [0]
    for i in range(1,n_jours):
        best_index = np.argmax((value_data[i]-value_data[i-1])/value_data[i-1])
        l_indices_achats_reg[best_index] += 10/value_data[i][best_index]
        if value_data is None:
            l_argent_reg.append(np.sum(l_indices_achats_reg*data[i+1]))
        else:
            l_argent_reg.append(np.sum(l_indices_achats_reg*value_data[i+1]))
    #Stratégie ML : acheter selon le modèle
    l_indices_achats_ML = np.zeros(n_indices)
    l_argent_ML = []
    # modeèle LSTM
    device = next(model_classifier.parameters()).device
    h0,c0 = torch.zeros((num_layers,hidden_dim), device=device), torch.zeros((num_layers,hidden_dim), device=device)
    for i in range(data_init.shape[0]):
        inp = data_init[i]
        if not isinstance(inp, torch.Tensor):
            inp = torch.tensor(inp, dtype=torch.float32, device=device)
        else:
            inp = inp.to(device)
        if inp.dim() == 1:
            inp = inp.unsqueeze(0)
        _,(h0,c0) = model_classifier(inp, (h0,c0))
        
    for i in range(n_jours-1):
        inp = data[i]
        if not isinstance(inp, torch.Tensor):
            inp = torch.tensor(inp, dtype=torch.float32, device=device)
        else:
            inp = inp.to(device)
        if inp.dim() == 1:
            inp = inp.unsqueeze(0)
            
        output, (h0,c0) = model_classifier(inp,(h0,c0))
        predicted_index = torch.argmax(output).item()
        l_indices_achats_ML[predicted_index] += 10/value_data[i][predicted_index]
        if value_data is None:
            l_argent_ML.append(np.sum(l_indices_achats_ML*data[i+1]))
        else:
            l_argent_ML.append(np.sum(l_indices_achats_ML*value_data[i+1]))

    l_indices_achats_ML_balanced = np.zeros(n_indices)
    l_argent_ML_balanced = []
    # modeèle LSTM
    device = next(model_classifier.parameters()).device
    h0,c0 = torch.zeros((num_layers,hidden_dim), device=device), torch.zeros((num_layers,hidden_dim), device=device)
    for i in range(data_init.shape[0]):
        inp = data_init[i]
        if not isinstance(inp, torch.Tensor):
            inp = torch.tensor(inp, dtype=torch.float32, device=device)
        else:
            inp = inp.to(device)
        if inp.dim() == 1:
            inp = inp.unsqueeze(0)
        _,(h0,c0) = model_classifier(inp, (h0,c0))
        
    for i in range(n_jours-1):
        inp = data[i]
        if not isinstance(inp, torch.Tensor):
            inp = torch.tensor(inp, dtype=torch.float32, device=device)
        else:
            inp = inp.to(device)
        if inp.dim() == 1:
            inp = inp.unsqueeze(0)
            
        output, (h0,c0) = model_classifier(inp,(h0,c0))
        predicted_index = torch.argmax(output).item()
        l_indices_achats_ML_balanced[predicted_index] += 10/value_data[i][predicted_index]
        # rebalancing 1% of the total portfolio every day by value
        total_value = np.sum(l_indices_achats_ML_balanced*value_data[i])
        for j in range(n_indices):
            current_value = l_indices_achats_ML_balanced[j]*value_data[i][j]
            desired_value = total_value / n_indices
            difference = desired_value - current_value
            l_indices_achats_ML_balanced[j] += (0.01 * difference) / value_data[i][j]
        if value_data is None:
            l_argent_ML_balanced.append(np.sum(l_indices_achats_ML_balanced*data[i+1]))
        else:
            l_argent_ML_balanced.append(np.sum(l_indices_achats_ML_balanced*value_data[i+1]))
    plt.figure(figsize=(12,6))
    plt.plot(l_argent_naif, label='Stratégie Naïve')
    plt.plot(l_argent_reg, label='Stratégie Régresseur')
    plt.plot(l_argent_ML, label='Stratégie ML')
    plt.plot(l_argent_ML_balanced, label='Stratégie ML Balanced')
    plt.xlabel('Jours')
    plt.ylabel('Argent accumulé')
    plt.title('Comparaison des stratégies de trading')
    plt.legend()
    plt.show()

def load_model(path_model):
    '''
    Charge un modèle enregistré à partir du chemin spécifié.
    '''
    model = torch.load(path_model)
    model.eval()
    return model

def trading_test_transformer(data_init, data, model_classifier, context_size, value_init=None, value_data=None):
    '''
    Test des stratégies de trading avec le modèle Transformer
    '''
    device = next(model_classifier.parameters()).device
    model_classifier.eval()
    
    n_jours = data.shape[0]
    n_indices = value_data.shape[1]
    
    # Stratégie naïve : on achète le plus gros indice chaque jour
    l_indices_achats_naif = np.zeros(n_indices)
    l_argent_naif = []
    for i in range(n_jours - 1):
        best_index = np.argmax(value_data[i])
        l_indices_achats_naif[best_index] += 10 / value_data[i][best_index]
        l_argent_naif.append(np.sum(l_indices_achats_naif * value_data[i + 1]))
    
    # Stratégie régresseur : acheter la plus grosse croissance
    l_indices_achats_reg = np.zeros(n_indices)
    l_argent_reg = [0]
    for i in range(1, n_jours):
        best_index = np.argmax((value_data[i] - value_data[i - 1]) / value_data[i - 1])
        l_indices_achats_reg[best_index] += 10 / value_data[i][best_index]
        l_argent_reg.append(np.sum(l_indices_achats_reg * value_data[i + 1]))
    
    # Stratégie ML : acheter selon le modèle
    l_indices_achats_ML = np.zeros(n_indices)
    l_argent_ML = []
    
    with torch.no_grad():
        for i in range(n_jours - 1):
            # Construire la séquence d'entrée
            if i < context_size:
                # Remplir avec les données d'entraînement au début
                inp = np.concatenate((data_init[-(context_size - i):], data[:i]), axis=0)
            else:
                inp = data[i - context_size:i]
            
            inp = torch.tensor(inp, dtype=torch.float32, device=device).unsqueeze(0)  # [1, context_size, n_features]
            
            output = model_classifier(inp)  # [1, context_size, n_classes]
            
            # Prédiction basée sur le dernier timestep
            predicted_index = torch.argmax(output[0, -1]).item()
            
            l_indices_achats_ML[predicted_index] += 10 / value_data[i][predicted_index]
            l_argent_ML.append(np.sum(l_indices_achats_ML * value_data[i + 1]))
    
    # Stratégie ML avec rebalancing
    l_indices_achats_ML_balanced = np.zeros(n_indices)
    l_argent_ML_balanced = []
    
    with torch.no_grad():
        for i in range(n_jours - 1):
            if i < context_size:
                inp = np.concatenate((data_init[-(context_size - i):], data[:i]), axis=0)
            else:
                inp = data[i - context_size:i]
            
            inp = torch.tensor(inp, dtype=torch.float32, device=device).unsqueeze(0)
            
            output = model_classifier(inp)
            predicted_index = torch.argmax(output[0, -1]).item()
            
            l_indices_achats_ML_balanced[predicted_index] += 10 / value_data[i][predicted_index]
            
            # Rebalancing
            total_value = np.sum(l_indices_achats_ML_balanced * value_data[i])
            for j in range(n_indices):
                current_value = l_indices_achats_ML_balanced[j] * value_data[i][j]
                desired_value = total_value / n_indices
                difference = desired_value - current_value
                l_indices_achats_ML_balanced[j] += (0.8 * difference) / value_data[i][j]
            
            l_argent_ML_balanced.append(np.sum(l_indices_achats_ML_balanced * value_data[i + 1]))
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(l_argent_naif, label='Stratégie Naïve')
    plt.plot(l_argent_reg, label='Stratégie Régresseur')
    plt.plot(l_argent_ML, label='Stratégie ML')
    plt.plot(l_argent_ML_balanced, label='Stratégie ML Balanced')
    plt.xlabel('Jours')
    plt.ylabel('Argent accumulé')
    plt.title('Comparaison des stratégies de trading')
    plt.legend()
    plt.show()
