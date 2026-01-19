import pandas as pd

df = pd.read_csv('table.csv')
# Filter jurisdictions where percent for > 70%
count_above_70 = df[df['percent for'] > 70.0].shape[0]
print(f"Final Answer: {count_above_70}")