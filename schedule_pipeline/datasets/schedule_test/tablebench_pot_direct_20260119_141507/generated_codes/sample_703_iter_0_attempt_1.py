import pandas as pd

df = pd.read_csv('table.csv')

# Extract the 'built' column and find min and max years
build_years = df['built'].astype(int)
earliest_year = build_years.min()
latest_year = build_years.max()
year_difference = latest_year - earliest_year

# Find the locomotive names corresponding to earliest and latest build years
earliest_locomotive = df.loc[df['built'] == earliest_year, 'locomotive'].values[0]
latest_locomotive = df.loc[df['built'] == latest_year, 'locomotive'].values[0]

print(f"Final Answer: {earliest_locomotive}, {latest_locomotive}, {year_difference}")