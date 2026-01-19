import pandas as pd

df = pd.read_csv('table.csv')
# Convert Start and End to integers
df['Start'] = pd.to_numeric(df['Start'])
df['End'] = pd.to_numeric(df['End'])

# Filter for ambassadors between 1956 and 1969
period1 = df[(df['Start'] >= 1956) & (df['End'] <= 1969)]
avg_tenure1 = (period1['End'] - period1['Start']).mean()

# Filter for ambassadors between 1981 and 2001
period2 = df[(df['Start'] >= 1981) & (df['End'] <= 2001)]
avg_tenure2 = (period2['End'] - period2['Start']).mean()

# Calculate the difference
difference = avg_tenure1 - avg_tenure2
print(f"Final Answer: {difference:.1f}")