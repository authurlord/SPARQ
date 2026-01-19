import pandas as pd

df = pd.read_csv('table.csv')
# Filter the row for 'guangdong' and get the value in the '2008' column
ranking_2008 = df[df['year'] == 'guangdong']['2008'].iloc[0]
print(f"Final Answer: {ranking_2008}")