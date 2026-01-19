import pandas as pd

df = pd.read_csv('table.csv')
# Filter the row for 'Black' voters and get the value for 1960
black_1960_vote = df[df['Unnamed: 0'] == 'Black']['1960'].values[0]
print(f"Final Answer: {black_1960_vote}")