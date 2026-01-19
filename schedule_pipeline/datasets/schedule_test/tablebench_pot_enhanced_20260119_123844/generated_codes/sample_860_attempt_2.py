import pandas as pd

df = pd.read_csv('table.csv')
# Extract executions in persona and effigie for Córdoba
cordoba_executions = df[df['Tribunal'] == 'Córdoba']['Executions in persona'].iloc[0]
cordoba_effigie = df[df['Tribunal'] == 'Córdoba']['Executions in effigie'].iloc[0]
cordoba_total = int(cordoba_executions) + int(cordoba_effigie)

# Extract executions in persona and effigie for Valencia
valencia_executions = df[df['Tribunal'] == 'Valencia']['Executions in persona'].iloc[0]
valencia_effigie = df[df['Tribunal'] == 'Valencia']['Executions in effigie'].iloc[0]
valencia_total = int(valencia_executions) + int(valencia_effigie)

# Calculate the difference
difference = cordoba_total - valencia_total
print(f"Final Answer: {difference}")