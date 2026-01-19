import pandas as pd

df = pd.read_csv('table.csv')
# Filter the row where Gold is 5 and Silver is 2
result = df[(df['Gold'] == '5') & (df['Silver'] == '2')]
# Extract the nation name
nation = result['Nation'].values[0]
print(f"Final Answer: {nation}")