import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where industry is 'banking'
banking_df = df[df['industry'] == 'banking']

# Take the top 5 companies (first 5 rows) in banking industry
top_5_banking = banking_df.head(5)

# Calculate the average revenue (in millions)
average_revenue = top_5_banking['revenue (millions)'].mean()
print(f"Final Answer: {average_revenue:.1f}")