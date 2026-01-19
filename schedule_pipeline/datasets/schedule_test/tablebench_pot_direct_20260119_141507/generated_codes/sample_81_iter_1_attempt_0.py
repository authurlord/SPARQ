import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# The 'bello' column is at index 1
# We want to find the ward where 'bello' == 66
ward_with_bello_66 = df[df['bello'] == 66]['ward'].values[0]

print(f"Final Answer: {ward_with_bello_66}")