import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Find the row for year 2004
row_2004 = df[df['year'] == '2004']

# Extract the average finish position for 2004
forecasted_avg_finish = row_2004['avg finish'].values[0]

print(f"Final Answer: {forecasted_avg_finish}")