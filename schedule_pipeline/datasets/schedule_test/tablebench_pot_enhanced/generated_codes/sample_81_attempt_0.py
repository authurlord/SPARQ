import pandas as pd

df = pd.read_csv('table.csv')
# Filter the row where 'bello' is 66
ward = df[df['bello'] == '66']['ward'].iloc[0]
print(f"Final Answer: {ward}")