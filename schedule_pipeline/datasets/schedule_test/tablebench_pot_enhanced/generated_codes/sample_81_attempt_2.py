import pandas as pd

df = pd.read_csv('table.csv')
# Filter the row where 'bello' equals 66
ward_with_66 = df[df['bello'] == '66']['ward'].values[0]
print(f"Final Answer: {ward_with_66}")