import pandas as pd

df = pd.read_csv('table.csv')
# Filter the row where Gold is 4 and Silver is 3
result = df[(df['Gold'] == '4') & (df['Silver'] == '3')]
nation = result['Nation'].iloc[0]
print(f"Final Answer: {nation}")