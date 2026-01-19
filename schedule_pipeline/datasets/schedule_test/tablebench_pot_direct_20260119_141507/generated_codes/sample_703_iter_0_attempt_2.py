import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'built' column to integers for numerical comparison
df['built'] = pd.to_numeric(df['built'], errors='coerce')
# Find the earliest and latest build years
earliest_year = df['built'].idxmin()
latest_year = df['built'].idxmax()
earliest_locomotive = df.loc[earliest_year, 'locomotive']
latest_locomotive = df.loc[latest_year, 'locomotive']
difference = df['built'].max() - df['built'].min()

print(f"Final Answer: {earliest_locomotive}, {difference}")