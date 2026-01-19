import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'district - wide' to numeric
df['district - wide'] = pd.to_numeric(df['district - wide'])

# Calculate the annual change (difference between consecutive years)
annual_changes = df['district - wide'].diff().dropna()

# Calculate the average annual change
average_change = annual_changes.mean()

print(f"Final Answer: {average_change:.1f}")