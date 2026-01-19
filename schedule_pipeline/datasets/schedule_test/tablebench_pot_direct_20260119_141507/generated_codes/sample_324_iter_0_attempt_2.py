import pandas as pd

df = pd.read_csv('table.csv')
# Filter the row for 2005
year_2005 = df[df['Year'] == '2005']
winner_2005 = year_2005['Winners'].iloc[0]
runner_up_2005 = year_2005['Runners-up'].iloc[0]
print(f"Final Answer: {winner_2005}, {runner_up_2005}")