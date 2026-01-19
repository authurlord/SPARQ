import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where Division is 'Third Division North'
third_div_north = df[df['Division'] == 'Third Division North']
# Extract the 'Goals' column (last column, index 8)
goals = third_div_north['Goals']
# Compute variance
variance = goals.var()
print(f"Final Answer: {variance:.1f}")