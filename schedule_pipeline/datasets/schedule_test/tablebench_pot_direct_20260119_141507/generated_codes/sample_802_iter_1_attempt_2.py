import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'rank' to integer type
df['rank'] = df['rank'].astype(int)
# Filter performers with rank >= 10
filtered_televotes = df[df['rank'] >= 10]['televotes'].astype(int)
# Compute the median
median_televotes = filtered_televotes.median()
print(f"Final Answer: {median_televotes}")