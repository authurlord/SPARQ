import pandas as pd

df = pd.read_csv('table.csv')
# Find the value for 'Black' in the column '1960'
black_voters_1960 = df.loc[df['Unnamed: 0'] == 'Black', '1960'].values[0]
print(f"Final Answer: {black_voters_1960}")