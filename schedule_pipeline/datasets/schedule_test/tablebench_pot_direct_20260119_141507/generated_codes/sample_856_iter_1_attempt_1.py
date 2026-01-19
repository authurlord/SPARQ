import pandas as pd

df = pd.read_csv('table.csv')

# Filter rows for 2004 election and 1999 election
row_2004 = df[df['event'] == '2004 election']
row_1999 = df[df['event'] == '1999 election']

# Extract 'dp / da' values, convert to int if possible, otherwise skip
dp_da_2004 = int(row_2004['dp / da'].values[0]) if row_2004['dp / da'].values[0] != '-' else 0
dp_da_1999 = int(row_1999['dp / da'].values[0]) if row_1999['dp / da'].values[0] != '-' else 0

# Calculate the difference
difference = dp_da_2004 - dp_da_1999
print(f"Final Answer: {difference}")