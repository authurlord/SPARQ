import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Political Rights' to numeric, coercing errors to NaN
df['Political Rights'] = pd.to_numeric(df['Political Rights'], errors='coerce')

# Filter data from 1975 to 1990 inclusive
filtered_df = df[(df['Year'] >= '1975') & (df['Year'] <= '1990')]

# Compute standard deviation of 'Political Rights' in the filtered data
std_political_rights = filtered_df['Political Rights'].std()

print(f"Final Answer: {std_political_rights:.2f}")