import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'year' to integer and compute average issue price per year
df['year'] = pd.to_numeric(df['year'], errors='coerce')
avg_issue_price_by_year = df.groupby('year')['issue price'].mean()

# Calculate the annual increase (difference between consecutive years)
annual_increases = avg_issue_price_by_year.diff().dropna()

# Compute the average of the annual increases
average_annual_increase = annual_increases.mean()

print(f"Final Answer: {average_annual_increase:.2f}")