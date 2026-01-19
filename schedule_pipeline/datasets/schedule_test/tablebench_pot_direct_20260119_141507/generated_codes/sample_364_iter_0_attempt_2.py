import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where 'martyred' is in the 19th century (1801 to 1900)
nineteenth_century_martyrs = df[(df['martyred'].astype(int) >= 1801) & (df['martyred'].astype(int) <= 1900)]
count_martyrs = len(nineteenth_century_martyrs)
print(f"Final Answer: {count_martyrs}")