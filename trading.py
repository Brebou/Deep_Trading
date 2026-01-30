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
        best_index = np.argmax(value_data[i])
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
    # Stratégie économe : conserver l'argent initial
    l_argent_conserve = np.arange(n_jours)*10
    # Stratégie uniforme : acheter des fractions égales de chaque indice au départ
    l_indices_achats_uniforme = np.zeros(n_indices)
    l_argent_uniforme = []
    for i in range(n_jours - 1):
        for j in range(n_indices):
            l_indices_achats_uniforme[j] += (10 / n_indices) / value_data[i][j]
        l_argent_uniforme.append(np.sum(l_indices_achats_uniforme * value_data[i + 1]))
    # Stratégie Uniforme Value : acheter des fractions égales de chaque indice pondérées par leur valeur au départ
    l_indices_achats_uniforme_value = np.zeros(n_indices)
    l_argent_uniforme_value = []
    for i in range(n_jours - 1):
        total_value = np.sum(value_data[i])
        for j in range(n_indices):
            proportion = value_data[i][j] / total_value
            l_indices_achats_uniforme_value[j] += (10 * proportion) / value_data[i][j]
        l_argent_uniforme_value.append(np.sum(l_indices_achats_uniforme_value * value_data[i + 1]))

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
    l_probas_ML = np.zeros((n_jours, n_indices))
    with torch.no_grad():
        for i in range(n_jours - 1):
            # Construire la séquence d'entrée
            if i < context_size:
                # Remplir avec les données d'entraînement au début
                inp = np.concatenate((data_init[-(context_size - i):], data[:i]), axis=0)
            else:
                inp = data[i - context_size:i]
            
            inp = torch.tensor(inp, dtype=torch.float32, device=device).unsqueeze(0)  # [1, context_size, n_features]
            
            output = model_classifier(inp)
            probas = torch.softmax(output[0, -1], dim=-1).cpu().numpy()
            l_probas_ML[i] = probas
            
            # Prédiction basée sur le dernier timestep
            predicted_index = torch.argmax(output[0, -1]).item()
            
            l_indices_achats_ML[predicted_index] += 10 / value_data[i][predicted_index]
            l_argent_ML.append(np.sum(l_indices_achats_ML * value_data[i + 1]))
    plt.figure(figsize=(12, 6))
    # Stratégie ML avec rebalancing usiant les probabilités pour k%
    k_list = [0.1]
    for k in k_list:
        l_indices_achats_ML_balanced = np.zeros(n_indices)
        l_argent_ML_balanced = []
        with torch.no_grad():
            for i in range(n_jours - 1):
                predicted_index = np.argmax(l_probas_ML[i])
                l_indices_achats_ML_balanced[predicted_index] += 10 / value_data[i][predicted_index]
                # rebalancing 1% of the total portfolio every day by value
                total_value = np.sum(l_indices_achats_ML_balanced * value_data[i])
                for j in range(n_indices):
                    current_value = l_indices_achats_ML_balanced[j] * value_data[i][j]
                    desired_value = total_value / n_indices
                    difference = desired_value - current_value
                    l_indices_achats_ML_balanced[j] += (k * difference) / value_data[i][j]
                l_argent_ML_balanced.append(np.sum(l_indices_achats_ML_balanced * value_data[i + 1]))
        plt.plot(l_argent_ML_balanced, label=f'Stratégie ML Balanced k={k}')
    # sampling k stratégies aléatoires sur les probas ML
    """
    k = 50
    for j in range(k):
        l_argent_random = []
        l_indices_achats_random = np.zeros(n_indices)
        for i in range(n_jours - 1):
            #sampling according to probas 
            index = np.random.choice(n_indices, p=  l_probas_ML[i]/np.sum(l_probas_ML[i]))
            l_indices_achats_random[index] += 10 / value_data[i][index]
            l_argent_random.append(np.sum(l_indices_achats_random * value_data[i + 1]))
        plt.plot(l_argent_random, color='gray', alpha=0.3)
    """
    # Plot
    plt.plot(l_argent_naif, label='Stratégie Naïve')
    plt.plot(l_argent_conserve, label='Stratégie Conserver')
    plt.plot(l_argent_uniforme, label='Stratégie Uniforme')
    plt.plot(l_argent_reg, label='Stratégie Régresseur')
    plt.plot(l_argent_uniforme_value, label='Stratégie Uniforme Value')
    plt.plot(l_argent_ML, label='Stratégie ML')
    plt.xlabel('Jours')
    plt.ylabel('Argent accumulé')
    plt.title('Comparaison des stratégies de trading')
    plt.legend()
    plt.show()

def trading_test_transformer_elec(data_init, data, model_classifier, context_size, value_init=None, value_data=None):
    '''
    Test des stratégies de trading avec le modèle Transformer pour les données d'électricité
    '''
    device = next(model_classifier.parameters()).device
    model_classifier.eval()
    
    n_jours = data.shape[0]
    
    somme_départ = 100
    l_probas_ML = np.zeros(n_jours)
    with torch.no_grad():
        for i in range(n_jours - 1):
            if i < context_size:
                inp = np.concatenate((data_init[-(context_size - i):], data[:i]), axis=0)
            else:
                inp = data[i - context_size:i]
            
            inp = torch.tensor(inp, dtype=torch.float32, device=device).unsqueeze(0)
            
            output = model_classifier(inp)
            probas = torch.softmax(output[0, -1], dim=-1).cpu().numpy()
            
            l_probas_ML[i + 1] = probas[1]

    # Trading basé sur argmax des probabilités
    l_argent_ML = [somme_départ]
    argent_ML = somme_départ
    elec_ML = 0
    for i in range(1, n_jours):
        # Acheter pour la moitié de l'argent si prédiction de hausse
        if l_probas_ML[i] > 0.5:
            achat = (argent_ML / 2) / value_data[i]
            elec_ML += achat
            argent_ML -= achat * value_data[i]
        else:
            # Vendre la moitié de l'électricité détenue
            vente = elec_ML / 2
            elec_ML -= vente
            argent_ML += vente * value_data[i]
        l_argent_ML.append(argent_ML + elec_ML * value_data[i])
    k = 10
    # sampling k stratégies aléatoires sur les probas ML
    for j in range(k):
        l_argent_random = [somme_départ]
        argent_random = somme_départ
        elec_random = 0
        for i in range(1, n_jours):
            if np.random.rand() < l_probas_ML[i]:
                achat = (argent_random / 2) / value_data[i]
                elec_random += achat
                argent_random -= achat * value_data[i]
            else:
                vente = elec_random / 2
                elec_random -= vente
                argent_random += vente * value_data[i]
            l_argent_random.append(argent_random + elec_random * value_data[i])
        plt.plot(l_argent_random, color='gray', alpha=0.3)
    # Stratégie naïve : acheter et garder
    l_elec_achats_naif = somme_départ / value_data[0]
    l_argent_naif = []
    for i in range(n_jours):
        l_argent_naif.append(l_elec_achats_naif * value_data[i])   
    # Stratégie régresseur : acheter quand la croissance est positive vendre sinon
    l_elec_achats_reg = 0
    argent_reg = somme_départ
    l_argent_reg = []
    for i in range(1, n_jours):
        croissance = (value_data[i] - value_data[i - 1])
        if croissance > 0:
            achat = (argent_reg / 2) / value_data[i]
            l_elec_achats_reg += achat
            argent_reg -= achat * value_data[i]
        else:
            vente = l_elec_achats_reg / 2
            l_elec_achats_reg -= vente
            argent_reg += vente * value_data[i]
        l_argent_reg.append(argent_reg + l_elec_achats_reg * value_data[i])
    # Stratégie pessimiste : vendre après une montée et acheter après une descente
    l_elec_achats_pess = 0
    argent_pess = somme_départ
    l_argent_pess = []
    for i in range(1, n_jours):
        croissance = (value_data[i] - value_data[i - 1])
        if croissance < 0:
            achat = (argent_pess / 2) / value_data[i]
            l_elec_achats_pess += achat
            argent_pess -= achat * value_data[i]
        else:
            vente = l_elec_achats_pess / 2
            l_elec_achats_pess -= vente
            argent_pess += vente * value_data[i]
        l_argent_pess.append(argent_pess + l_elec_achats_pess * value_data[i])
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(l_argent_naif, label='Stratégie Naïve', color='blue')
    plt.plot(l_argent_reg, label='Stratégie Régresseur', color='green')
    plt.plot(l_argent_pess, label='Stratégie Pessimiste', color='red')
    plt.plot(l_argent_ML, label='Stratégie ML', color='orange')
    plt.xlabel('Jours')
    plt.ylabel('Argent accumulé')
    plt.title('Comparaison des stratégies de trading sur données d\'électricité')
    plt.legend()
    plt.show()




