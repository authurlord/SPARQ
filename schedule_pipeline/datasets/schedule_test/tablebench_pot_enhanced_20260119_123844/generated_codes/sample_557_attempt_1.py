import pandas as pd

df = pd.read_csv('table.csv')
# Convert columns to appropriate types
df['Year'] = df['Year'].astype(int)
df['Conservative councillors'] = df['Conservative councillors'].astype(int)

# Calculate the difference in Conservative councillors from the previous year
df['diff'] = df['Conservative councillors'].diff()

# Find the year with the maximum increase
max_increase_year = df.loc[df['diff'].idxmax(), 'Year']

print(f"Final Answer: {max_increase_year}")