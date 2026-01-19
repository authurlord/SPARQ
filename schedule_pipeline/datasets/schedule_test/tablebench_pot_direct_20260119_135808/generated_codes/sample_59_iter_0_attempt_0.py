import pandas as pd

df = pd.read_csv('table.csv')
# Filter the row where Rank is 4 and Gold is 4
nation = df[(df['Rank'] == '4') & (df['Gold'] == '4')]['Nation'].iloc[0]
print(f"Final Answer: {nation}")