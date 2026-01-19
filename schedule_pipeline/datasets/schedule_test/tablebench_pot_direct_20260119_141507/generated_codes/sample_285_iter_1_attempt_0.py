import pandas as pd

df = pd.read_csv('table.csv')

# Filter for banking industry
banking_df = df[df['industry'] == 'banking']

# Get the top 5 companies in banking (first 5 rows)
top_5_banking = banking_df.head(5)

# Calculate average revenue of these top 5 companies
average_revenue = top_5_banking['revenue (millions)'].astype(float).mean()

print(f"Final Answer: {average_revenue:.1f}")