import pandas as pd

df = pd.read_csv('table.csv')
# Filter the row for llanhilleth rfc
llanhilleth_points = df[df['club'] == 'llanhilleth rfc']['points'].iloc[0]
print(f"Final Answer: {llanhilleth_points}")