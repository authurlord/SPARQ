import pandas as pd

df = pd.read_csv('table.csv')
# Count countries with at least one semifinalist
count_semifinalists = (df['semifinalists'] > 0).sum()
print(f"Final Answer: {count_semifinalists}")