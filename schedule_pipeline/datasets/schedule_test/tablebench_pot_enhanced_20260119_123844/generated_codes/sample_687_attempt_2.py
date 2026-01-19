import pandas as pd

df = pd.read_csv('table.csv')
# Find the year with the highest crude birth rate
max_birth_rate_year = df.loc[df['Crude birth rate (per 1000)'].idxmax(), 'Unnamed: 0']
print(f"Final Answer: {max_birth_rate_year}")