import pandas as pd

df = pd.read_csv('table.csv')
# Filter the data for the year 2005
year_2005 = df[df['Year'] == '2005']
# Extract winner and runner-up
winner = year_2005['Winners'].values[0]
runner_up = year_2005['Runners-up'].values[0]
print(f"Final Answer: {winner}, {runner_up}")