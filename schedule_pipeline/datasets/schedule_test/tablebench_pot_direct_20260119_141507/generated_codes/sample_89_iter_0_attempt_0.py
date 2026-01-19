import pandas as pd

df = pd.read_csv('table.csv')
# Find the row where 'year' is 'guangdong' and get the value in the '2008' column
gongdong_rank_2008 = df.loc[df['year'] == 'guangdong', '2008'].values[0]
print(f"Final Answer: {gongdong_rank_2008}")