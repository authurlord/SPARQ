import pandas as pd

df = pd.read_csv('table.csv')
# Find the value for 'Black' in the '1960' column
black_1960 = df.loc[df['Unnamed: 0'] == 'Black', '1960'].values[0]
print(f"Final Answer: {black_1960}")