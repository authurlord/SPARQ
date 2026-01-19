import pandas as pd

df = pd.read_csv('table.csv')
# Extract 'executions in persona' column and clean data (remove percentage part)
executions_persona = df['executions in persona'].str.replace(r'\s*\([^)]+\)', '', regex=True).astype(int)
# Calculate median
median_executions = executions_persona.median()
print(f"Final Answer: {median_executions}")