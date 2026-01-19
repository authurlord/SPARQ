import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Start' and 'End' to integers
df['Start'] = pd.to_numeric(df['Start'])
df['End'] = pd.to_numeric(df['End'])

# Filter for ambassadors serving between 1956 and 1969
group1 = df[(df['Start'] >= 1956) & (df['End'] <= 1969)]
tenure1 = group1['End'] - group1['Start']
avg_tenant1 = tenure1.mean()

# Filter for ambassadors serving between 1981 and 2001
group2 = df[(df['Start'] >= 1981) & (df['End'] <= 2001)]
tenure2 = group2['End'] - group2['Start']
avg_tenant2 = tenure2.mean()

# Calculate how much shorter the first average tenure was
difference = avg_tenant1 - avg_tenant2

print(f"Final Answer: {difference:.1f}")