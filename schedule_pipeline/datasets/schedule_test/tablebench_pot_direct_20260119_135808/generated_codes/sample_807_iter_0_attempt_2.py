import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where Division is 'Third Division North'
third_division_north = df[df['Division'] == 'Third Division North']
# Extract the 'Goals' column (from the 'Goals' column under 'League')
goals = third_division_north['Goals'].astype(float)
# Calculate variance
variance = goals.var()
print(f"Final Answer: {variance:.2f}")