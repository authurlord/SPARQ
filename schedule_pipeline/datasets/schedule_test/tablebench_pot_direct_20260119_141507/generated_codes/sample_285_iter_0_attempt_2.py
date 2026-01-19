import pandas as pd

df = pd.read_csv('table.csv')
# Filter for banking industry
banking_df = df[df['industry'] == 'banking']

# Sort by revenue (descending) and take top 5
top_5_banking = banking_df.nlargest(5, 'revenue (millions)')

# Calculate average revenue of these top 5
avg_revenue = top_5_banking['revenue (millions)'].mean()
print(f"Final Answer: {avg_revenue:.1f}")