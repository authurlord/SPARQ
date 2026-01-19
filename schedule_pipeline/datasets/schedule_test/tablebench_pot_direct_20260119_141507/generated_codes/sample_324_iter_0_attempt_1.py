import pandas as pd

df = pd.read_csv('table.csv')
# Filter the row for year 2005
year_2005 = df[df['Year'] == '2005']
winner_2005 = year_2005['Winners'].values[0]
runner_up_2005 = year_2005['Runners-up'].values[0]
print(f"Final Answer: {winner_2005}, {runner_up_2005}")