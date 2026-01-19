import pandas as pd

df = pd.read_csv('table.csv')
# Convert '2008' and '2009' columns to numeric
df['2008'] = pd.to_numeric(df['2008'])
df['2009'] = pd.to_numeric(df['2009'])

# Calculate growth rate from 2008 to 2009
df['growth_rate'] = (df['2009'] - df['2008']) / df['2008'] * 100

# Find the airport with the highest growth rate
max_growth_airport = df.loc[df['growth_rate'].idxmax()]['airport']

print(f"Final Answer: {max_growth_airport}")