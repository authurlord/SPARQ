import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where Club is 'Liverpool' and the League is 'Premier League'
# The "League" column is at index 3, and "Goals" in Premier League is at index 5
filtered_df = df[(df['Club'] == 'Liverpool') & (df['League'] == 'Premier League')]
# Sum the goals from the "Goals" column in "League_1" (index 5)
total_goals = filtered_df['League_1'].sum()
print(f"Final Answer: {total_goals}")