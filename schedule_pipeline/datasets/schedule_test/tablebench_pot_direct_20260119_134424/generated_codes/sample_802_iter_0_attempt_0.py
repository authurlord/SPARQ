import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where rank is >= 10
filtered_df = df[df['rank'].astype(int) >= 10]
# Convert televotes to integer and calculate median
median_televotes = filtered_df['televotes'].astype(int).median()
print(f"Final Answer: {median_televotes}")