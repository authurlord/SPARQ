import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'issue price' to float and extract the prices by year
issue_prices = df['issue price'].astype(float)
years = df['year'].astype(int)

# Calculate the yearly increases
price_increases = []
for i in range(1, len(issue_prices)):
    increase = issue_prices[i] - issue_prices[i-1]
    price_increases.append(increase)

# Compute the average increase
average_increase = sum(price_increases) / len(price_increases)
print(f"Final Answer: {average_increase:.2f}")