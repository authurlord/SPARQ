import pandas as pd

df = pd.read_csv('table.csv')
# Filter the row where total is 6 and get the nation name
nation_with_total_6 = df[df['total'] == 6]['nation'].values[0]
print(f"Final Answer: {nation_with_total_6}")