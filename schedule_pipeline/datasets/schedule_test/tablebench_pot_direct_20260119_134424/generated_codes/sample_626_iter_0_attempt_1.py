import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where 'Year built' is 1909 and count the number of entries
num_1909 = df[df['Year built'] == '1909'].shape[0]
print(f"Final Answer: {num_1909}")