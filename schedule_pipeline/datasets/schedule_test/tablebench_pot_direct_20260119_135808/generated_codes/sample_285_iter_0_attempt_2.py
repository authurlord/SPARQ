import pandas as pd

df = pd.read_csv('table.csv')
# Filter for banking industry
banking_companies = df[df['industry'] == 'banking']
# Sort by rank to get the top 5
top_5_bankers = banking_companies.nsmallest(5, 'rank')
# Calculate average revenue
avg_revenue = top_5_bankers['revenue (millions)'].astype(float).mean()
print(f"Final Answer: {avg_revenue:.1f}")