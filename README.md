# DeepTrading

**DeepTrading** is a Python project that uses LSTM neural networks to predict stock prices and automate investment decisions.

## 📌 Description

This project aims to:
- **Predict** stock price trends using LSTM models.
- **Automate** investment of a fixed amount of money based on model predictions.
- **Optimize** trading strategies using deep learning.

## 🛠 Installation

To run this project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/Brebou/Deep_Trading/tree/classification
   ```
2. To train the model, run the following command
```   
   python3 deeptrading.py --dataset_path data/stocks.csv --lr 0.0002 --save_dir results --nepochs 10 --hidden_dim 128 --num_layers 2
```


