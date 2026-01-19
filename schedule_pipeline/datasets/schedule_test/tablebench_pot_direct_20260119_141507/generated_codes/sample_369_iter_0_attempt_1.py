import pandas as pd

df = pd.read_csv('table.csv')
# Count countries with at least one semifinalist (semifinalists > 0)
count_semi = df[df['semifinalists'] > 0].shape[0]
print(f"Final Answer: {count_semi}")