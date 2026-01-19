import pandas as pd

df = pd.read_csv('table.csv')

# Identify Spanish-related language categories
spanish_columns = ['spanish', 'only spanish', 'native and spanish']

# Sum Spanish speakers per municipality
df_spanish = df[spanish_columns].apply(pd.to_numeric, errors='coerce').sum(axis=1)

# Find the municipality with the highest total Spanish speakers
max_municipality = df.iloc[df_spanish.idxmax(), 0]
print(f"Final Answer: {max_municipality}")