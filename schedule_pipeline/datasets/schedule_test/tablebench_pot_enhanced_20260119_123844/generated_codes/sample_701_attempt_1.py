import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Injuries (US $000)' to numeric, coercing errors to NaN
df['Injuries (US $000)'] = pd.to_numeric(df['Injuries (US $000)'], errors='coerce')

# Drop rows where injuries are NaN
df_clean = df.dropna(subset=['Injuries (US $000)'])

# Find the year with the maximum injuries
max_injuries_year = df_clean.loc[df_clean['Injuries (US $000)'].idxmax(), 'Year']
print(f"Final Answer: {max_injuries_year}")