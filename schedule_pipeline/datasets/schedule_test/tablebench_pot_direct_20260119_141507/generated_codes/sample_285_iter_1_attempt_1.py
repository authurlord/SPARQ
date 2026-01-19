import pandas as pd

df = pd.read_csv('table.csv')
# Convert revenue column to numeric to avoid parsing errors
df['revenue (millions)'] = pd.to_numeric(df['revenue (millions)'], errors='coerce')

# Filter for banking industry
banking_df = df[df['industry'] == 'banking']

# Select top 5 companies (by rank) in banking
top_5_banking = banking_df.head(5)

# Calculate average revenue of these top 5
average_revenue = top_5_banking['revenue (millions)'].mean()

print(f"Final Answer: {average_revenue:.1f}")