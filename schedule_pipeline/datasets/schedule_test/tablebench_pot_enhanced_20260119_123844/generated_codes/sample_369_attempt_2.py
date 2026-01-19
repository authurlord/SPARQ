import pandas as pd

df = pd.read_csv('table.csv')
# Count countries with at least one semifinalist
count_semifinalists = df[df['semifinalists'] > 0].shape[0]
print(f"Final Answer: {count_semifinalists}")