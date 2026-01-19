import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where Division is 'Third Division North'
third_division_north = df[df['Division'] == 'Third Division North']
# Extract the 'Goals' column from the 'League' section (column index 5)
goals = third_division_north['Goals']
# Calculate variance
variance = goals.astype(float).var()
print(f"Final Answer: {variance:.2f}")