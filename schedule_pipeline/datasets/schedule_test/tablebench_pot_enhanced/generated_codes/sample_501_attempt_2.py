import pandas as pd

df = pd.read_csv('table.csv')

# Drop rows where either 21st or 22nd ionization energy is missing
df_clean = df.dropna(subset=['21st', '22nd'])

# Convert '21st' and '22nd' columns to numeric
df_clean['21st'] = pd.to_numeric(df_clean['21st'])
df_clean['22nd'] = pd.to_numeric(df_clean['22nd'])

# Calculate the ratio of 22nd to 21st ionization energy
df_clean['ratio'] = df_clean['22nd'] / df_clean['21st']

# Find the element with the highest ratio
max_ratio_row = df_clean.loc[df_clean['ratio'].idxmax()]

# Output the symbol of the element with the highest ratio
print(f"Final Answer: {max_ratio_row['symbol']}")