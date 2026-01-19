import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows for Córdoba and Valencia
cordoba_executions = df[df['Tribunal'] == 'Córdoba']['Executions in persona'].values[0] + df[df['Tribunal'] == 'Córdoba']['Executions in effigie'].values[0]
valencia_executions = df[df['Tribunal'] == 'Valencia']['Executions in persona'].values[0] + df[df['Tribunal'] == 'Valencia']['Executions in effigie'].values[0]
# Calculate the difference
difference = cordoba_executions - valencia_executions
print(f"Final Answer: {difference}")