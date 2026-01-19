import pandas as pd

df = pd.read_csv('table.csv')
# Remove the 'Total' and 'Source' rows
df_clean = df[df['Party'] != 'Total']
df_clean = df_clean[df_clean['Party'] != 'Source: http://www.stat.go.jp/data/chouki/27.htm']
# Convert '%' column to float
df_clean['%'] = pd.to_numeric(df_clean['%'])
# Count parties with more than 10% votes
count_parties = (df_clean['%'] > 10).sum()
print(f"Final Answer: {count_parties}")