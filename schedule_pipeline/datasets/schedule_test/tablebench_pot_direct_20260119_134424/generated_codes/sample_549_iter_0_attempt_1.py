import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'district - wide' to integer for numerical operations
df['district - wide'] = df['district - wide'].astype(int)

# Calculate the annual change (difference between consecutive years)
annual_changes = df['district - wide'].diff().dropna()

# Calculate the average annual change
average_change = annual_changes.mean()

print(f"Final Answer: {average_change:.1f}")