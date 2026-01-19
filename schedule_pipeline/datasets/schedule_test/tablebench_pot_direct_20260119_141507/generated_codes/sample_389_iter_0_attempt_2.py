import pandas as pd

df = pd.read_csv('table.csv')
# Count the number of cantons with 'yes' in the 'separate secondary schools' column
yes_count = df[df['separate secondary schools'] == 'yes'].shape[0]
print(f"Final Answer: {yes_count}")