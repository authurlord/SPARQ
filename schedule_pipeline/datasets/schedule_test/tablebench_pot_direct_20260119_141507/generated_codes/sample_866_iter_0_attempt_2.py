import pandas as pd

df = pd.read_csv('table.csv')

# Convert Start and End to integers
df['Start'] = pd.to_numeric(df['Start'], errors='coerce')
df['End'] = pd.to_numeric(df['End'], errors='coerce')

# Filter ambassador records only
df_ambassador = df[df['Title'] == 'Ambassador'].copy()

# Calculate tenure (End - Start)
df_ambassador['tenure'] = df_ambassador['End'] - df_ambassador['Start']

# Filter for period 1956 to 1969
period1 = df_ambassador[(df_ambassador['Start'] >= 1956) & (df_ambassador['End'] <= 1969)]

# Filter for period 1981 to 2001
period2 = df_ambassador[(df_ambassador['Start'] >= 1981) & (df_ambassador['End'] <= 2001)]

# Calculate average tenure for each group
avg_tenure_period1 = period1['tenure'].mean()
avg_tenure_period2 = period2['tenure'].mean()

# Compute the difference
difference = avg_tenure_period1 - avg_tenure_period2

print(f"Final Answer: {difference:.1f}")