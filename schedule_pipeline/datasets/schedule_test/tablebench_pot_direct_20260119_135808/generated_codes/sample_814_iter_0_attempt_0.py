import pandas as pd

df = pd.read_csv('table.csv')
# Clean 'executions in persona' column: extract integer part before parentheses
df['executions in persona'] = df['executions in persona'].str.replace(r'\s*\(.*\)', '', regex=True)
df['executions in persona'] = pd.to_numeric(df['executions in persona'], errors='coerce')

# Drop the 'total' row as it's a summary
df_clean = df[df['tribunal'] != 'total']

# Calculate median
median_executions = df_clean['executions in persona'].median()
print(f"Final Answer: {median_executions}")