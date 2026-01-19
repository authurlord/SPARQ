import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where rank >= 10
filtered_df = df[df['rank'].astype(int) >= 10]
# Convert televotes to integers
filtered_df['televotes'] = filtered_df['televotes'].astype(int)
# Calculate median
median_televotes = filtered_df['televotes'].median()
print(f"Final Answer: {median_televotes}")