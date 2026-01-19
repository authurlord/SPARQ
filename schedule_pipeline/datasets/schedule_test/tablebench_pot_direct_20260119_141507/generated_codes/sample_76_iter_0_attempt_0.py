import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where Gold is 4 and Silver is 3
result = df[(df['Gold'] == 4) & (df['Silver'] == 3)]
# Extract the nation name
nation = result.iloc[0]['Nation'] if not result.empty else None
print(f"Final Answer: {nation}")