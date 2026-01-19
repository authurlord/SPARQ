import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows for 1999 election and 2004 election
row_1999 = df[df['event'] == '1999 election']
row_2004 = df[df['event'] == '2004 election']

# Extract dp/da values
dp_da_1999 = int(row_1999['dp / da'].values[0])
dp_da_2004 = int(row_2004['dp / da'].values[0])

# Calculate difference
difference = dp_da_2004 - dp_da_1999
print(f"Final Answer: {difference}")