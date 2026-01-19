import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'revenue (millions)' to numeric, coercing errors to NaN
df['revenue (millions)'] = pd.to_numeric(df['revenue (millions)'], errors='coerce')
# Filter for 'banking' industry
banking_companies = df[df['industry'] == 'banking']
# Sort by revenue in descending order and get top 5
top_5_bankers = banking_companies.nlargest(5, 'revenue (millions)')
# Calculate average revenue
avg_revenue = top_5_bankers['revenue (millions)'].mean()
print(f"Final Answer: {avg_revenue:.1f}")