import pandas as pd

df = pd.read_csv('table.csv')
# Extract 'executions in persona' column and convert to integer, excluding the total row
executions_persona = df['executions in persona'].iloc[:-1]  # Exclude the 'total' row
executions_persona = executions_persona.astype(int)
median_executions = executions_persona.median()
print(f"Final Answer: {median_executions}")