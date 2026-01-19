import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where rank >= 10
filtered_df = df[df['rank'] >= 10]
# Calculate median of televotes
median_televotes = filtered_df['televotes'].median()
print(f"Final Answer: {median_televotes}")