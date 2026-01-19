import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'total revenue' to integers
df['total revenue'] = df['total revenue'].astype(int)

# Extract revenue for 2001 and 2002
revenue_2001 = df[df['year'] == '2001']['total revenue'].values[0]
revenue_2002 = df[df['year'] == '2002']['total revenue'].values[0]

# Calculate growth rate
growth_rate = ((revenue_2002 - revenue_2001) / revenue_2001) * 100
print(f"Final Answer: {growth_rate:.2f}")