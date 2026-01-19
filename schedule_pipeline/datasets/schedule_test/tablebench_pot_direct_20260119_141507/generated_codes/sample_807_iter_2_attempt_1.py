import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Filter rows where Division is "Third Division North"
third_div_north = df[df['Division'] == 'Third Division North']

# Extract the 'Goals' column (index 4)
goals = third_div_north.iloc[:, 4]  # Column index 4 is 'Goals'

# Compute variance of goals
variance = goals.var()

print(f"Final Answer: {variance:.1f}")