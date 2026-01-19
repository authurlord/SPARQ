import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where '1 credit' is 100 or more
filtered_rows = df[df['1 credit'].astype(int) >= 100]
# Count the number of such rows
count = len(filtered_rows)
print(f"Final Answer: {count}")