import pandas as pd
df = pd.read_csv('table.csv')
df_1990s = df[(df['Year'] >= 1990) & (df['Year'] <= 1999)]
winners_1990s = df_1990s[df_1990s['Outcome'] == 'Winner']