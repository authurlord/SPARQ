import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where Division is 'Third Division North'
third_division_north = df[df['Division'] == 'Third Division North']
# Extract the 'Goals' column and calculate variance
variance_goals = third_division_north['Goals'].var()
print(f"Final Answer: {variance_goals:.2f}")