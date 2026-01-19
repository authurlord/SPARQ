import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where Division is 'Third Division North' and not a total row
filtered_df = df[(df['Division'] == 'Third Division North') & (~df['Season'].str.contains('Total'))]
# Extract the 'Goals' column
goals = filtered_df['Goals']
# Calculate variance
variance = goals.var()
print(f"Final Answer: {variance:.1f}")