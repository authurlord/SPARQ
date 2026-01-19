import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'district - wide' to numeric
df['district - wide'] = pd.to_numeric(df['district - wide'])

# Calculate the annual change
annual_changes = df['district - wide'].diff().dropna()

# Compute the average annual change
avg_change = annual_changes.mean()

print(f"Final Answer: {avg_change:.1f}")