import pandas as pd

df = pd.read_csv('table.csv')
# Sort by year to ensure chronological order
df = df.sort_values(by='year')
# Calculate the change in total revenue from one year to the next
df['revenue_change'] = df['total revenue'].diff()

# Determine the trend based on the sign of the changes
trend = "increasing" if df['revenue_change'].gt(0).all() else "decreasing" if df['revenue_change'].lt(0).all() else "fluctuating"
print(f"Final Answer: {trend}")