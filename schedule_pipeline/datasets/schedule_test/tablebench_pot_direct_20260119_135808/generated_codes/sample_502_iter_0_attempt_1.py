import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'total revenue' to numeric
df['total revenue'] = pd.to_numeric(df['total revenue'])

# Extract total revenue for 2001 and 2002
revenue_2001 = df[df['year'] == '2001']['total revenue'].iloc[0]
revenue_2002 = df[df['year'] == '2002']['total revenue'].iloc[0]

# Calculate growth rate
growth_rate = ((revenue_2002 - revenue_2001) / revenue_2001) * 100

print(f"Final Answer: {growth_rate:.1f}")