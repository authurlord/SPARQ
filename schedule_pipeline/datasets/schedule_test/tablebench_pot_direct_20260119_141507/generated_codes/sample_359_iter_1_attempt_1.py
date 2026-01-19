import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Convert '1 credit' column to numeric, handling any non-numeric values
df['1 credit'] = pd.to_numeric(df['1 credit'], errors='coerce')

# Filter rows where '1 credit' >= 100 and exclude the last row (theoretical return)
# The last row is 'theoretical return', so we slice up to that row
filtered_rows = df.iloc[:-1][df['1 credit'] >= 100]

# Count the number of such rows
count = len(filtered_rows)

print(f"Final Answer: {count}")