import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where occurrence > 1
filtered_df = df[df['occurrence'] > 1]
# Count the number of such rows
count = len(filtered_df)
print(f"Final Answer: {count}")