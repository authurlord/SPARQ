import pandas as pd

df = pd.read_csv('table.csv')
# Calculate overseas rate as (overseas / total usaaf) * 100
df['overseas_rate'] = (df['overseas'] / df['total usaaf']) * 100

# Find the date with the highest overseas rate
max_rate_row = df.loc[df['overseas_rate'].idxmax()]
highest_overseas_rate_date = max_rate_row['date']

print(f"Final Answer: {highest_overseas_rate_date}")