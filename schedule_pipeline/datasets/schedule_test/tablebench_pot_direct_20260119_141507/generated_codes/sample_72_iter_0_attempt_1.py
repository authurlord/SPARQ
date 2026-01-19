import pandas as pd

df = pd.read_csv('table.csv')
# Find the ward where 'bello' column has value 51
ward_with_bello_51 = df[df['bello'] == '51']['ward'].values[0]
print(f"Final Answer: {ward_with_bello_51}")