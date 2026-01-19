import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where Division is 'Third Division North'
third_div_north_goals = df[df['Division'] == 'Third Division North']['Goals']
# Compute variance of goals
variance_goals = third_div_north_goals.var()
print(f"Final Answer: {variance_goals:.1f}")