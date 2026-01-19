import pandas as pd

df = pd.read_csv('table.csv')

# Filter rows where Division is 'Third Division North'
third_div_north = df[df['Division'] == 'Third Division North']

# Extract the 'Goals' column (note: there are two 'Goals' columns; we take the second one)
# The second 'Goals' column is labeled as 'Goals' in the row structure
goals_values = third_div_north['Goals'].dropna()

# Calculate variance
variance = goals_values.var()

print(f"Final Answer: {variance:.1f}")