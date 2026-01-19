import pandas as pd

df = pd.read_csv('table.csv')
# Find the ward where 'bello' equals 66
ward_with_bello_66 = df[df['bello'] == '66']['ward'].iloc[0]
print(f"Final Answer: {ward_with_bello_66}")