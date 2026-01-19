import pandas as pd

df = pd.read_csv('table.csv')
# Drop the 'total' row and extract 'executions in persona' values
executions_persona = df[df['tribunal'] != 'total']['executions in persona'].str.replace(r'\(.*\)', '', regex=True).astype(int)
# Calculate median
median_executions = executions_persona.median()
print(f"Final Answer: {median_executions}")