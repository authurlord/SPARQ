import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where Gold == 3 and Total == 8
nation = df[(df['Gold'] == 3) & (df['Total'] == 8)]['Nation'].values[0]
print(f"Final Answer: {nation}")