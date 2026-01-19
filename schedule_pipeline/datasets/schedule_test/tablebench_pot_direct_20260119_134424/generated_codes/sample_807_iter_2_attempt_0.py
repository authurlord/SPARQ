import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where Division is 'Third Division North'
third_division_north = df[df['Division'] == 'Third Division North']
# Extract the 'Goals' column (index 4, which corresponds to 'Goals' in the data)
goals = third_division_north['Goals'].astype(float)
# Calculate variance
variance = goals.var()
print(f"Final Answer: {variance:.4f}")