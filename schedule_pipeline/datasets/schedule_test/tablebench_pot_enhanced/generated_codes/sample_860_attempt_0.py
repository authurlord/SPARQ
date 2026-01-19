import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows for Córdoba and Valencia
cordoba_executions = df[df['Tribunal'] == 'Córdoba'][['Executions in persona', 'Executions in effigie']].sum().sum()
valencia_executions = df[df['Tribunal'] == 'Valencia'][['Executions in persona', 'Executions in effigie']].sum().sum()
# Calculate the difference
difference = cordoba_executions - valencia_executions
print(f"Final Answer: {difference}")