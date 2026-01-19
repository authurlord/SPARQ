import pandas as pd

df = pd.read_csv('table.csv')
# Filter for banking industry
banking_companies = df[df['industry'] == 'banking']
# Take top 5 by rank
top_5_bankers = banking_companies.head(5)
# Calculate average revenue
avg_revenue = top_5_bankers['revenue (millions)'].mean()
print(f"Final Answer: {avg_revenue:.1f}")