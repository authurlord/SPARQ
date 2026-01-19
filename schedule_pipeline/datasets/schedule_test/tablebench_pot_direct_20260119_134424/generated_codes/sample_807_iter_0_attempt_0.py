import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where Division is 'Third Division North'
third_division_north = df[df['Division'] == 'Third Division North']

# Extract goals scored in the league (column 'Goals' under 'League')
goals = third_division_north['Goals'].astype(int)

# Calculate variance
variance = goals.var()

print(f"Final Answer: {variance:.2f}")