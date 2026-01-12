import yfinance as yf
import pandas as pd

# Liste des 30 meilleurs indices boursiers pour le trading par IA
best_indices = [
    # Indices américains (5)
    "^GSPC",      # S&P 500
    "^NDX",       # Nasdaq 100
    "^DJI",       # Dow Jones Industrial Average
    "^RUT",       # Russell 2000
    "^VIX",       # S&P 500 VIX (Volatilité)
    
    # Indices européens (9)
    "^GDAXI",     # DAX 40 (Allemagne)
    "^FTSE",      # FTSE 100 (Royaume-Uni)
    "^FCHI",      # CAC 40 (France)
    "^STOXX50E",  # Euro Stoxx 50
    "^IBEX",      # IBEX 35 (Espagne)
    "FTSEMIB.MI", # FTSE MIB (Italie)
    "^SSMI",      # SMI (Suisse)
    "^AEX",       # AEX (Pays-Bas)
    "^BFX",       # BEL 20 (Belgique)
    
    # Indices asiatiques (8)
    "^N225",      # Nikkei 225 (Japon)
    "^HSI",       # Hang Seng (Hong Kong)
    "^NSEI",      # NIFTY 50 (Inde)
    "000001.SS",  # Shanghai Composite (Chine)
    "^KS11",      # Kospi (Corée du Sud)
    "^TWII",      # Taiwan Weighted (Taiwan)
    "^STI",       # Straits Times (Singapour)
    "^AXJO",      # ASX 200 (Australie)
    
    # Indices émergents et autres (4)
    "^BVSP",      # Bovespa (Brésil)
    "^MXX",       # IPC Mexico (Mexique)
    "EEM",        # iShares MSCI Emerging Markets ETF (Marchés émergents globaux)
    "^GSPTSE",    # S&P TSX Composite (Canada)
]
n_stocks = len(best_indices)

print('Number of stocks :', n_stocks)

# Télécharger toutes les données d'abord
all_data = {}
for index in best_indices:
    ticker = yf.Ticker(index)
    data = ticker.history(period="18y")
    data.reset_index(inplace=True)
    # Convertir en date uniquement (sans heure ni timezone)
    data['Date'] = pd.to_datetime(data['Date']).dt.date
    all_data[index] = data
    print(f'{index}: {len(data)} jours')

# Trouver les dates communes
common_dates = set(all_data[best_indices[0]]['Date'])
for index in best_indices[1:]:
    common_dates = common_dates.intersection(set(all_data[index]['Date']))

common_dates = sorted(list(common_dates))
print(f"\nDates communes: {len(common_dates)}")
if common_dates:
    print(f"Première date commune: {common_dates[0]}")
    print(f"Dernière date commune: {common_dates[-1]}")
# Filtrer chaque indice sur les dates communes
aligned_data = {}
for index in best_indices:
    df = all_data[index]
    df_aligned = df[df['Date'].isin(common_dates)].copy()
    df_aligned = df_aligned.sort_values('Date').reset_index(drop=True)
    aligned_data[index] = df_aligned
    print(f'{index} après alignement: {len(df_aligned)} jours')
# Nettoyer les données et créer le dictionnaire final
df_stock_cleaned = {}
for index in best_indices:
    data = aligned_data[index].drop(columns=['Date'])
    # cleaning the data in each column (making it an float)
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    df_stock_cleaned[index] = data

for i,index in enumerate(df_stock_cleaned.keys()):
    if i==0:
        res = df_stock_cleaned[index]
        continue
    else:
        res = pd.concat([res, df_stock_cleaned[index]], axis=1)

print(res.shape)
print(set(res.columns))
save_adress = './data/stocks.csv'

res.to_csv(save_adress, index = False)

# Computation of the growth
df = pd.read_csv(save_adress)

# Removing zero-columns and computing growth
df.drop(columns = ['Capital Gains', 'Dividends', 'Stock Splits'], inplace = True)
for i in range(1, n_stocks):
    df.drop(columns = [f'Dividends.{i}', f'Stock Splits.{i}'], inplace = True)
    df[f'Growth.{i}'] = df[f'Close.{i}'].pct_change()

# Computing growth of close price
df['Growth'] = df['Close'].pct_change()
for i in range(1, n_stocks):
    df[f'Growth.{i}'] = df[f'Close.{i}'].pct_change()

# Removing NaN value
df = df.fillna(0)

df.to_csv(save_adress, index = False)

