import pandas as pd

df = pd.read_csv('table.csv')
# Count nations with at least one gold medal
gold_count = df[df['Gold'] > 0].shape[0]
print(f"Final Answer: {gold_count}")