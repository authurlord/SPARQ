import pandas as pd

df = pd.read_csv('table.csv')
# Filter for Third Division North
third_division_north = df[df['Division'] == 'Third Division North']
# Extract goals scored in league (column 'Goals' under 'League')
goals = third_division_north['Goals'].astype(float)
# Calculate variance
variance = goals.var()
print(f"Final Answer: {variance:.2f}")