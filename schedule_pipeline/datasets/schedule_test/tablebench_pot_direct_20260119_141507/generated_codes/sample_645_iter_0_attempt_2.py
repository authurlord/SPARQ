import pandas as pd

df = pd.read_csv('table.csv')
# Convert sales and profits to numeric
sales = pd.to_numeric(df['sales (billion )'], errors='coerce')
profits = pd.to_numeric(df['profits (billion )'], errors='coerce')

# Calculate correlation coefficient
correlation = sales.corr(profits)
print(f"Final Answer: {correlation:.2f}")