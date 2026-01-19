import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where Division is 'Third Division North'
third_div_north = df[df['Division'] == 'Third Division North']
# Extract the 'Goals' column (the last 'Goals' column in the data)
goals = third_div_north['Goals'].dropna()
# Calculate variance
variance = goals.var()
print(f"Final Answer: {variance:.2f}")