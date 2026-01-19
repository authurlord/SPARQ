import pandas as pd

df = pd.read_csv('table.csv')
# Filter the row where Year is 2005
row_2005 = df[df['Year'] == '2005']
# Extract winners and runners-up
winner = row_2005['Winners'].values[0]
runner_up = row_2005['Runners-up'].values[0]
print(f"Final Answer: {winner}, {runner_up}")