import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'district - wide' to integers
df['district - wide'] = df['district - wide'].astype(int)
# Calculate year-over-year changes
changes = df['district - wide'].diff().dropna()
# Calculate average annual change
avg_change = changes.mean()
print(f"Final Answer: {avg_change:.1f}")