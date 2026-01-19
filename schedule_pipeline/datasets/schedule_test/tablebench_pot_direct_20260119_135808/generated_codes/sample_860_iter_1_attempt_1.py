import pandas as pd

df = pd.read_csv('table.csv')

# Filter rows for Córdoba and Valencia
cordoba_row = df[df['Tribunal'] == 'Córdoba']
valencia_row = df[df['Tribunal'] == 'Valencia']

# Extract and convert values to integers
cordoba_executions = int(cordoba_row['Executions in persona'].values[0]) + int(cordoba_row['Executions in effigie'].values[0])
valencia_executions = int(valencia_row['Executions in persona'].values[0]) + int(valencia_row['Executions in effigie'].values[0])

# Calculate the difference
difference = cordoba_executions - valencia_executions

print(f"Final Answer: {difference}")