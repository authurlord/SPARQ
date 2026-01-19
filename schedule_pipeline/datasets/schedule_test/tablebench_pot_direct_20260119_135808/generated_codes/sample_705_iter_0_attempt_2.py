import pandas as pd

df = pd.read_csv('table.csv')
# Drop the first row which contains headers
df = df.drop(0)
# Convert the 'Copper (mg)' column to numeric, handling '-' as NaN
df['Copper (mg)'] = pd.to_numeric(df['Copper (mg)'], errors='coerce')
# Find the staple with the highest Copper (mg) value
max_copper_index = df['Copper (mg)'].idxmax()
staple_with_max_copper = df.loc[max_copper_index, 'STAPLE:']
print(f"Final Answer: {staple_with_max_copper}")