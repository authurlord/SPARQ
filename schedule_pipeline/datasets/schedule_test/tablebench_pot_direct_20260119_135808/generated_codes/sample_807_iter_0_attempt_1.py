import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where Division is 'Third Division North'
third_division_north = df[df['Division'] == 'Third Division North']
# Extract the 'Goals' column (from 'Goals' in 'Apps' and 'Goals' columns)
goals = third_division_north['Goals'].astype(int)
# Calculate variance
variance_goals = goals.var()
print(f"Final Answer: {variance_goals:.1f}")