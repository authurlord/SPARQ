import pandas as pd

df = pd.read_csv('table.csv')
# Drop the first row which contains headers
df = df.drop(0)
# Convert 'Copper (mg)' column to numeric, handling '-' as NaN
df['Copper (mg)'] = pd.to_numeric(df['Copper (mg)'], errors='coerce')
# Find the staple with the highest Copper (mg)
max_copper_staple = df.loc[df['Copper (mg)'].idxmax(), 'STAPLE:']
print(f"Final Answer: {max_copper_staple}")