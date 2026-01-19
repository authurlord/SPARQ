import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows for 2004 election and 1999 election
dp_da_2004 = df[df['event'] == '2004 election']['dp / da'].values[0]
dp_da_1999 = df[df['event'] == '1999 election']['dp / da'].values[0]
# Calculate the difference
difference = int(dp_da_2004) - int(dp_da_1999)
print(f"Final Answer: {difference}")