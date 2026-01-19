import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'district - wide' to integer for numerical operations
df['district - wide'] = df['district - wide'].astype(int)
# Calculate annual changes
annual_changes = df['district - wide'].diff().dropna()
# Calculate average annual change
avg_change = annual_changes.mean()
print(f"Final Answer: {avg_change:.1f}")