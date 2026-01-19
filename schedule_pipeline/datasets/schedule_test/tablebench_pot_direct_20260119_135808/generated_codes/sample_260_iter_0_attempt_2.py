import pandas as pd

df = pd.read_csv('table.csv')

# Extract year from 'Date Built' and 'Date Withdrawn'
df['Year Built'] = df['Date Built'].str.extract(r'(\d{4})').astype(int)
df['Year Withdrawn'] = df['Date Withdrawn'].str.extract(r'(\d{4})').astype(int)

# Filter for locomotives built in 1938
locomotives_1938 = df[df['Year Built'] == 1938]

# Calculate years in service
locomotives_1938['Years in Service'] = locomotives_1938['Year Withdrawn'] - locomotives_1938['Year Built']

# Find maximum years in service
max_years = locomotives_1938['Years in Service'].max()

print(f"Final Answer: {max_years}")