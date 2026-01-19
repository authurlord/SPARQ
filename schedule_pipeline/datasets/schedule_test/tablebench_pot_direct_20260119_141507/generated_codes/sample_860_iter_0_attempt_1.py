import pandas as pd

df = pd.read_csv('table.csv')
# Extract the relevant values for Córdoba and Valencia
cordoba_executions = df.loc[df['Tribunal'] == 'Córdoba', ['Executions in persona', 'Executions in effigie']].values[0]
valencia_executions = df.loc[df['Tribunal'] == 'Valencia', ['Executions in persona', 'Executions in effigie']].values[0]

total_cordoba = cordoba_executions[0] + cordoba_executions[1]
total_valencia = valencia_executions[0] + valencia_executions[1]

difference = total_cordoba - total_valencia
print(f"Final Answer: {difference}")