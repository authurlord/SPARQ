import pandas as pd

df = pd.read_csv('table.csv')
# Filter nation with 4 gold and 3 silver medals
result = df[(df['Gold'] == 4) & (df['Silver'] == 3)]
if not result.empty:
    print(f"Final Answer: {result.iloc[0]['Nation']}")
else:
    print("Final Answer: None")