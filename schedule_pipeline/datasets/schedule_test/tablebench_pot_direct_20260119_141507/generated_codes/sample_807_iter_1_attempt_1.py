import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where Division is 'Third Division North'
filtered_df = df[df['Division'] == 'Third Division North']
# Extract the 'Goals' column (index 4)
goals = filtered_df.iloc[:, 4]  # column index 4 is 'Goals'
# Compute variance
variance = goals.var()
print(f"Final Answer: {variance:.2f}")