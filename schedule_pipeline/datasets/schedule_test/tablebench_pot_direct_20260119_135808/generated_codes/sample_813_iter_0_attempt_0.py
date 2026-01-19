import pandas as pd

df = pd.read_csv('table.csv')

# Remove the 'TOTAL' row
df = df[df['Party'] != 'TOTAL']

# Extract seats for First Duma and Fourth Duma
first_duma = df['First Duma'].str.extract('(\d+)').astype(int)
fourth_duma = df['Fourth Duma'].str.extract('(\d+)').astype(int)

# Filter out rows where either value is missing
valid_rows = first_duma.notna() & fourth_duma.notna()
first_duma = first_duma[valid_rows].values.flatten()
fourth_duma = fourth_duma[valid_rows].values.flatten()
parties = df[valid_rows]['Party'].values

# Calculate percentage increase
percentage_increase = ((fourth_duma - first_duma) / first_duma) * 100

# Find the party with the highest percentage increase
max_increase_idx = percentage_increase.argmax()
best_party = parties[max_increase_idx]

print(f"Final Answer: {best_party}")