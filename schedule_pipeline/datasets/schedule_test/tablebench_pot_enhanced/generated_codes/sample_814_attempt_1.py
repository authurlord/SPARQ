import pandas as pd

df = pd.read_csv('table.csv')
# Clean the 'executions in persona' column by removing percentage and parentheses
df['executions in persona'] = df['executions in persona'].str.replace(r'[()]', '', regex=True).str.replace('%', '', regex=True)
# Convert to integer
df['executions in persona'] = pd.to_numeric(df['executions in persona'], errors='coerce')
# Calculate median, excluding the 'total' row
median_executions = df[df['tribunal'] != 'total']['executions in persona'].median()
print(f"Final Answer: {median_executions}")