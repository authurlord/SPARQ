import pandas as pd

df = pd.read_csv('table.csv')
# Convert the necessary columns to numeric, coercing errors to NaN if needed
df['number of hurricanes'] = pd.to_numeric(df['number of hurricanes'], errors='coerce')
df['number of major hurricanes'] = pd.to_numeric(df['number of major hurricanes'], errors='coerce')
df['number of tropical storms'] = pd.to_numeric(df['number of tropical storms'], errors='coerce')

# Calculate HSI: (hurricanes + major hurricanes) / tropical storms
df['HSI'] = (df['number of hurricanes'] + df['number of major hurricanes']) / df['number of tropical storms']

# Find the year with the highest HSI
max_hsi_year = df.loc[df['HSI'].idxmax(), 'year']
print(f"Final Answer: {max_hsi_year}")