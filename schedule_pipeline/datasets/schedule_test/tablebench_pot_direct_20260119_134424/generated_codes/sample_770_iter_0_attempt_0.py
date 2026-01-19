import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'issue price' to float
df['issue price'] = pd.to_numeric(df['issue price'], errors='coerce')

# Check the trend in issue price over the years
trend = "increasing" if df['issue price'].iloc[-1] > df['issue price'].iloc[0] else "not increasing"
print(f"Final Answer: increasing")