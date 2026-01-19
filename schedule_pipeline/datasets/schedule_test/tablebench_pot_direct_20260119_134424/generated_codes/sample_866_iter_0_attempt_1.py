import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Start' and 'End' to integers
df['Start'] = df['Start'].astype(int)
df['End'] = df['End'].astype(int)

# Filter for the two time periods
period1 = df[(df['Start'] >= 1956) & (df['End'] <= 1969)]
period2 = df[(df['Start'] >= 1981) & (df['End'] <= 2001)]

# Calculate tenure in years
period1['tenure'] = period1['End'] - period1['Start']
period2['tenure'] = period2['End'] - period2['Start']

# Calculate average tenure
avg_tenure_period1 = period1['tenure'].mean()
avg_tenure_period2 = period2['tenure'].mean()

# Calculate the difference
difference = avg_tenure_period1 - avg_tenure_period2

print(f"Final Answer: {difference:.1f}")