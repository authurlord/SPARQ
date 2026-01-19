import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Year' to integer
df['Year'] = pd.to_numeric(df['Year'])
# Sort by Year
df = df.sort_values('Year')
# Calculate the difference in Conservative councillors
df['diff'] = df['Conservative councillors'].diff()
# Find the year with the maximum increase
max_increase_year = df.loc[df['diff'].idxmax(), 'Year']
print(f"Final Answer: {max_increase_year}")