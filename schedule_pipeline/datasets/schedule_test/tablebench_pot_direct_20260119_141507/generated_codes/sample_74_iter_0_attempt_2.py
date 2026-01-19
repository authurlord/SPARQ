import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where Gold == 3 and Total == 8
result = df[(df['Gold'] == 3) & (df['Total'] == 8)]
# Extract the Nation name
nation = result.iloc[0]['Nation']
print(f"Final Answer: {nation}")