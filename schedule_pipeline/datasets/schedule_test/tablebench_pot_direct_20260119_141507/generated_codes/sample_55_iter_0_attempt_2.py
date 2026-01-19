import pandas as pd

df = pd.read_csv('table.csv')
# Filter nation with 1 gold and 4 silver medals
result = df[(df['gold'] == 1) & (df['silver'] == 4)]
if not result.empty:
    print(f"Final Answer: {result['nation'].values[0]}")
else:
    print("Final Answer: none")