import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'district - wide' to integers
df['district - wide'] = df['district - wide'].astype(int)
# Calculate the annual change
annual_changes = df['district - wide'].diff().dropna()
# Calculate the average annual change
avg_annual_change = annual_changes.mean()
print(f"Final Answer: {avg_annual_change:.1f}")