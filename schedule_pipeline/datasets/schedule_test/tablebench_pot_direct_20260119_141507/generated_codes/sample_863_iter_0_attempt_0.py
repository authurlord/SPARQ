import pandas as pd

df = pd.read_csv('table.csv')
# Group by year and compute average issue price
avg_issue_price = df.groupby('year')['issue price'].mean()

# Find the year with the highest and lowest average issue price
max_year = avg_issue_price.idxmax()
min_year = avg_issue_price.idxmin()
max_price = avg_issue_price.max()
min_price = avg_issue_price.min()
difference = max_price - min_price

print(f"Final Answer: {max_year}, {difference:.2f}")