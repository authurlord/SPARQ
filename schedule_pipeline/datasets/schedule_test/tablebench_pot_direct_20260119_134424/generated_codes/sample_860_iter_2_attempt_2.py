import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for Córdoba and Valencia
cordoba_executions = df[df['Tribunal'] == 'Córdoba']['Executions in persona'].iloc[0] + df[df['Tribunal'] == 'Córdoba']['Executions in effigie'].iloc[0]
valencia_executions = df[df['Tribunal'] == 'Valencia']['Executions in persona'].iloc[0] + df[df['Tribunal'] == 'Valencia']['Executions in effigie'].iloc[0]
# Calculate the difference
difference = cordoba_executions - valencia_executions
print(f"Final Answer: {difference}")