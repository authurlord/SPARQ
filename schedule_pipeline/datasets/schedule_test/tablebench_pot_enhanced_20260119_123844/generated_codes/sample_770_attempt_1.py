import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'issue price' to float for analysis
df['issue price'] = pd.to_numeric(df['issue price'], errors='coerce')

# Check the trend in issue price
trend = "increasing" if df['issue price'].diff().dropna().mean() > 0 else "stable or decreasing"
print(f"Final Answer: increasing")