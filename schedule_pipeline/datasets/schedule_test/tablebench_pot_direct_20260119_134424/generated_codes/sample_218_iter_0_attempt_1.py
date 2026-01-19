import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('table.csv')
df['Level at Trent Bridge m'] = pd.to_numeric(df['Level at Trent Bridge m'], errors='coerce')
df['Peak Flow m3/s'] = pd.to_numeric(df['Peak Flow m3/s'].str.replace(',', ''), errors='coerce')
df = df.dropna(subset=['Rank'])
df_sorted = df.sort_values(by='Level at Trent Bridge m')
plt.figure(figsize=(10, 6))
plt.scatter(df_sorted['Level at Trent Bridge m'], df_sorted['Peak Flow m3/s'], color='blue', label='Peak Flow vs Water Level')