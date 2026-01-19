import pandas as pd

df = pd.read_csv('table.csv')
# Filter for teams in Third Division North
third_division_north = df[df['Division'] == 'Third Division North']
# Extract the 'Goals' column (from 'Goals' in the data, which corresponds to 'Goals' in the second row of the data)
goals = third_division_north['Goals'].astype(int)
# Calculate variance
variance = goals.var()
print(f"Final Answer: {variance:.2f}")