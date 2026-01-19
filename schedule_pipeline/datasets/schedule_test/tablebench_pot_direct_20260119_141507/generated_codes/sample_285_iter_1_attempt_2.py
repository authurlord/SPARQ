import pandas as pd

df = pd.read_csv('table.csv')
# Filter banking industry
banking_df = df[df['industry'] == 'banking'].copy()
# Sort by revenue (descending) and take top 5
top_5_banking = banking_df.sort_values(by='revenue (millions)', ascending=False).head(5)
# Calculate average revenue
average_revenue = top_5_banking['revenue (millions)'].mean()
print(f"Final Answer: {average_revenue:.1f}")