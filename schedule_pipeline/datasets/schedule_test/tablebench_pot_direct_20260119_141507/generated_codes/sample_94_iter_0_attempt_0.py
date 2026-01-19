import pandas as pd

df = pd.read_csv('table.csv')
# Filter the row for George McLean and get the league goals
george_mclean_goals = df[df['name'] == 'george mclean']['league goals'].values[0]
print(f"Final Answer: {george_mclean_goals}")