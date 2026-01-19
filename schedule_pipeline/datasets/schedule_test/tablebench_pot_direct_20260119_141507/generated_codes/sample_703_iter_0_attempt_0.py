import pandas as pd

df = pd.read_csv('table.csv')
# Extract the 'built' column and find the earliest and latest build years
build_years = df['built'].astype(int)
earliest_year = build_years.min()
latest_year = build_years.max()
year_difference = latest_year - earliest_year

# Find the locomotive built in the earliest year
earliest_locomotive = df[df['built'] == earliest_year]['locomotive'].iloc[0]

print(f"Final Answer: {earliest_locomotive}, {year_difference}")