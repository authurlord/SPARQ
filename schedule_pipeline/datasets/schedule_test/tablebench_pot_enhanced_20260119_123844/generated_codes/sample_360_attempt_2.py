import pandas as pd

df = pd.read_csv('table.csv')
# Count nations with at least one gold medal
nations_with_gold = df[df['Gold'] > '0'].shape[0]
print(f"Final Answer: {nations_with_gold}")