import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Convert Start and End to integers
df['Start'] = pd.to_numeric(df['Start'], errors='coerce')
df['End'] = pd.to_numeric(df['End'], errors='coerce')

# Filter for period 1: 1956 to 1969
period1 = df[(df['Start'] >= 1956) & (df['End'] <= 1969)]
# Filter for period 2: 1981 to 2001
period2 = df[(df['Start'] >= 1981) & (df['End'] <= 2001)]

# Calculate average tenure for each period
avg_tenure_period1 = period1['End'].astype(int) - period1['Start'].astype(int)
avg_tenure_period2 = period2['End'].astype(int) - period2['Start'].astype(int)

avg_tenure_1 = avg_tenure_period1.mean()
avg_tenure_2 = avg_tenure_period2.mean()

# Difference
difference = avg_tenure_2 - avg_tenure_1
print(f"Final Answer: {difference:.1f}")