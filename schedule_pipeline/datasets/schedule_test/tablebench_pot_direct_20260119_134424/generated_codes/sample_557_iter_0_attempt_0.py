import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Year' to integer and 'Conservative councillors' to numeric
df['Year'] = pd.to_numeric(df['Year'])
df['Conservative councillors'] = pd.to_numeric(df['Conservative councillors'])

# Calculate the difference in Conservative councillors between consecutive years
df['diff'] = df['Conservative councillors'].diff()

# Find the year with the maximum increase (excluding the first year)
max_increase_year = df.loc[df['diff'].idxmax(), 'Year']

print(f"Final Answer: {max_increase_year}")