import pandas as pd

df = pd.read_csv('table.csv')

# Filter rows where Division is "Third Division North" and not summary rows
filtered_df = df[(df['Division'] == 'Third Division North') & (~df['Season'].str.contains('Total|Career'))]

# Extract the 'Goals' column (index 4)
goals = filtered_df.iloc[:, 4].dropna()

# Calculate variance
variance = goals.var()

print(f"Final Answer: {variance:.2f}")