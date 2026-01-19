import pandas as pd

df = pd.read_csv('table.csv')
# Filter for martyred in the 19th century (1800-1899)
nineteenth_century_martyrs = df[df['martyred'].astype(int).between(1800, 1899)]
count = len(nineteenth_century_martyrs)
print(f"Final Answer: {count}")