import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where 'Date Built' contains '1938'
df_1938 = df[df['Date Built'].str.contains('1938')]
# Extract year from 'Date Withdrawn' by splitting and taking the last part
df_1938['Withdrawn Year'] = df_1938['Date Withdrawn'].str.split().str[-1]
# Convert to integer
df_1938['Withdrawn Year'] = pd.to_numeric(df_1938['Withdrawn Year'])
# Calculate years in service
df_1938['Years in Service'] = df_1938['Withdrawn Year'] - 1938
# Find maximum years in service
max_years = df_1938['Years in Service'].max()
print(f"Final Answer: {max_years}")