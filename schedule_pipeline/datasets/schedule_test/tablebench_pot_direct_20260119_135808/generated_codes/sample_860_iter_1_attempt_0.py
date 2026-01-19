import pandas as pd

df = pd.read_csv('table.csv')

# Filter data for Córdoba and Valencia
cordoba_row = df[df['Tribunal'] == 'Córdoba']
valencia_row = df[df['Tribunal'] == 'Valencia']

# Convert 'Executions in persona' and 'Executions in effigie' to integers, replacing '?' with 0
cordoba_persona = int(cordoba_row['Executions in persona'].values[0].replace('?', '0'))
cordoba_effigie = int(cordoba_row['Executions in effigie'].values[0].replace('?', '0'))
valencia_persona = int(valencia_row['Executions in persona'].values[0].replace('?', '0'))
valencia_effigie = int(valencia_row['Executions in effigie'].values[0].replace('?', '0'))

# Calculate total executions
cordoba_total = cordoba_persona + cordoba_effigie
valencia_total = valencia_persona + valencia_effigie

# Compute difference
difference = cordoba_total - valencia_total
print(f"Final Answer: {difference}")