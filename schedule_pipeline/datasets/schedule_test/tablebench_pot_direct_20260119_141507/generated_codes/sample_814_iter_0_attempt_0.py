import pandas as pd

df = pd.read_csv('table.csv')
# Remove the 'total' row as it is not a tribunal
df_filtered = df[df['tribunal'] != 'total']
# Extract the 'executions in persona' column and calculate median
median_executions = df_filtered['executions in persona'].median()
print(f"Final Answer: {median_executions}")