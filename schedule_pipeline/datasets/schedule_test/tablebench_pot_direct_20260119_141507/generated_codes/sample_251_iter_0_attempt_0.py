import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where Club is 'Lincoln City' and Division is 'Third Division North'
filtered = df[(df['Club'] == 'Lincoln City') & (df['Division'] == 'Third Division North')]
# Sum the 'Goals' column
total_goals = filtered['Goals'].sum()
print(f"Final Answer: {total_goals}")