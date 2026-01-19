import pandas as pd

df = pd.read_csv('table.csv')
# Find the ward where 'bello' column value is 66
ward_with_66_bello = df[df['bello'] == '66']['ward'].values[0]
print(f"Final Answer: {ward_with_66_bello}")