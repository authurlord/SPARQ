import pandas as pd

df = pd.read_csv('table.csv')

# Calculate HSI for each year
df['hs_i'] = (df['number of hurricanes'] + df['number of major hurricanes']) / df['number of tropical storms']

# Find the year with the highest HSI
max_hsi_year = df.loc[df['hs_i'].idxmax(), 'year']

print(f"Final Answer: {max_hsi_year}")