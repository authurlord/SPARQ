import pandas as pd

df = pd.read_csv('table.csv')

# Convert Start and End to integers
df['Start'] = pd.to_numeric(df['Start'])
df['End'] = pd.to_numeric(df['End'])

# Filter for 1956–1969
period1 = df[(df['Start'] >= 1956) & (df['Start'] <= 1969)]
tenure1 = period1['End'] - period1['Start']
avg_tenure1 = tenure1.mean()

# Filter for 1981–2001
period2 = df[(df['Start'] >= 1981) & (df['Start'] <= 2001)]
tenure2 = period2['End'] - period2['Start']
avg_tenure2 = tenure2.mean()

# Calculate the difference
difference = avg_tenure1 - avg_tenure2
print(f"Final Answer: {difference:.1f}")