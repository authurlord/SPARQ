import pandas as pd

df = pd.read_csv('table.csv')
# Filter the row for Black voters and get the value for 1960
black_voters_1960 = df[df['Unnamed: 0'] == 'Black']['1960'].iloc[0]
print(f"Final Answer: {black_voters_1960}")