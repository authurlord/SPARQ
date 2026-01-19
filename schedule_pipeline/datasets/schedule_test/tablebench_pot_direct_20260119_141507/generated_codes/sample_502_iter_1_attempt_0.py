import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'total revenue' to numeric, coercing errors to NaN if any
df['total revenue'] = pd.to_numeric(df['total revenue'], errors='coerce')

# Extract total revenue for 2001 and 2002
revenue_2001 = df.loc[df['year'] == '2001', 'total revenue'].values[0]
revenue_2002 = df.loc[df['year'] == '2002', 'total revenue'].values[0]

# Calculate revenue growth rate
growth_rate = ((revenue_2002 - revenue_2001) / revenue_2001) * 100

print(f"Final Answer: {growth_rate:.2f}")