import pandas as pd

df = pd.read_csv('table.csv')

# Find the 'dp / da' values for 2004 election and 1999 election
row_2004 = df[df['event'] == '2004 election']
row_1999 = df[df['event'] == '1999 election']

# Extract 'dp / da' values and convert to int, ignoring '-'
dp_da_2004 = int(row_2004['dp / da'].values[0]) if row_2004['dp / da'].values[0] != '-' else 0
dp_da_1999 = int(row_1999['dp / da'].values[0]) if row_1999['dp / da'].values[0] != '-' else 0

# Calculate the difference
difference = dp_da_2004 - dp_da_1999
print(f"Final Answer: {difference}")