import pandas as pd

df = pd.read_csv('table.csv')

# Filter rows where Club is "Lincoln City" and Division is "Third Division North"
filtered_rows = df[(df['Club'] == 'Lincoln City') & (df['Division'] == 'Third Division North')]

# Sum the 'Goals' column (index 4)
total_goals = filtered_rows['Goals'].sum()

print(f"Final Answer: {total_goals}")