import pandas as pd

df = pd.read_csv('table.csv')

# Convert Start and End to integers
df['Start'] = pd.to_numeric(df['Start'], errors='coerce')
df['End'] = pd.to_numeric(df['End'], errors='coerce')

# Filter for periods
period1 = df[(df['Start'] >= 1956) & (df['End'] <= 1969)]
period2 = df[(df['Start'] >= 1981) & (df['End'] <= 2001)]

# Calculate average tenure for each group
avg_tenure_1 = period1['End'].astype(int) - period1['Start'].astype(int)
avg_tenure_2 = period2['End'].astype(int) - period2['Start'].astype(int)

avg_tenure_1_mean = avg_tenure_1.mean()
avg_tenure_2_mean = avg_tenure_2.mean()

# Compute the difference
difference = avg_tenure_1_mean - avg_tenure_2_mean

print(f"Final Answer: {difference:.2f}")