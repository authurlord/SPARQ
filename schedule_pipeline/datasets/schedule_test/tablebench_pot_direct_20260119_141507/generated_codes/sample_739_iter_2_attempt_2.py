import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'viewers' to numeric, handling any parsing errors
df['viewers'] = pd.to_numeric(df['viewers'], errors='coerce')
# Drop any rows with missing viewers (in case of invalid entries)
df = df.dropna(subset=['viewers'])

# Calculate average viewership and average BBC Three weekly ranking
avg_viewers = df['viewers'].mean()
avg_rank = df['bbc three weekly ranking'].mean()

print(f"Final Answer: {avg_viewers:.0f}, {avg_rank:.0f}")