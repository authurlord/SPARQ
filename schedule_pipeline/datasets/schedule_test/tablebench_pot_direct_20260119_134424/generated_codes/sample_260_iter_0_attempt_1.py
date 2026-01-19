import pandas as pd

df = pd.read_csv('table.csv')

# Filter locomotives built in 1938
df_1938 = df[df['Date Built'].str.contains('1938')]

# Extract year from 'Date Built' and 'Date Withdrawn'
df_1938['Year Built'] = df_1938['Date Built'].str.extract(r'(\d{4})').astype(int)
df_1938['Year Withdrawn'] = df_1938['Date Withdrawn'].str.extract(r'(\d{4})').astype(int)

# Calculate years in service
df_1938['Years in Service'] = df_1938['Year Withdrawn'] - df_1938['Year Built']

# Find maximum years in service
max_years = df_1938['Years in Service'].max()

print(f"Final Answer: {max_years}")