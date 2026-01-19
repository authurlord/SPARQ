import pandas as pd

df = pd.read_csv('table.csv')
# Count the number of cantons with 'yes' in 'separate secondary schools'
count_yes = df[df['separate secondary schools'] == 'yes'].shape[0]
print(f"Final Answer: {count_yes}")