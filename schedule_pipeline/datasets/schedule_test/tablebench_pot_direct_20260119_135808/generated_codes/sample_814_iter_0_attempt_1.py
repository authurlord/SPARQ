import pandas as pd

df = pd.read_csv('table.csv')
# Clean 'executions in persona' column: remove percentage and extract numeric value
df['executions in persona'] = df['executions in persona'].str.replace(r'\(.*\)', '', regex=True).str.strip()
# Convert to integer
executions_persona = pd.to_numeric(df['executions in persona'], errors='coerce')
# Calculate median
median_executions = executions_persona.median()
print(f"Final Answer: {median_executions}")