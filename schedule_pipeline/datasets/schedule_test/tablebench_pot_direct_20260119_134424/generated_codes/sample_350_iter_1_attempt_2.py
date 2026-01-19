import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'total goals' to numeric, coercing errors to NaN
df['total goals'] = pd.to_numeric(df['total goals'], errors='coerce')
# Count players with total goals > 3
count = (df['total goals'] > 3).sum()
print(f"Final Answer: {count}")