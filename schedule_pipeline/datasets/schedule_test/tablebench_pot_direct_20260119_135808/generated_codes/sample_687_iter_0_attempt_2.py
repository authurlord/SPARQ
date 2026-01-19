import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Crude birth rate (per 1000)' to float, handling any potential formatting issues
df['Crude birth rate (per 1000)'] = pd.to_numeric(df['Crude birth rate (per 1000)'], errors='coerce')
# Find the year with the maximum crude birth rate
max_year = df.loc[df['Crude birth rate (per 1000)'].idxmax(), 'Unnamed: 0']
print(f"Final Answer: {max_year}")