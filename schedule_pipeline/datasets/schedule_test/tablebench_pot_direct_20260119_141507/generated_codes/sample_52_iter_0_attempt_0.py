import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where Gold is 5 and Silver is 2
result = df[(df['Gold'] == 5) & (df['Silver'] == 2)]
if not result.empty:
    nation = result.iloc[0]['Nation']
    print(f"Final Answer: {nation}")
else:
    print("Final Answer: None")