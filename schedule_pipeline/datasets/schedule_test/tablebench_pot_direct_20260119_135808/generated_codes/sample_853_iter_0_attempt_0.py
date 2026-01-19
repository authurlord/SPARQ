import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'US Chart position' to numeric, coercing errors to NaN
df['US Chart position'] = pd.to_numeric(df['US Chart position'], errors='coerce')

# Drop rows where 'US Chart position' is NaN (due to R&B labels)
df_clean = df.dropna(subset=['US Chart position'])

# Find the year with the highest US chart position (lowest number = best position)
max_position_year = df_clean.loc[df_clean['US Chart position'].idxmin(), 'Year']

# Find the year with the lowest US chart position (highest number = worst position)
min_position_year = df_clean.loc[df_clean['US Chart position'].idxmax(), 'Year']

print(f"Final Answer: {max_position_year}, {min_position_year}")