import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where occurrence > 1
count = df[df['occurrence'] > 1].shape[0]
print(f"Final Answer: {count}")