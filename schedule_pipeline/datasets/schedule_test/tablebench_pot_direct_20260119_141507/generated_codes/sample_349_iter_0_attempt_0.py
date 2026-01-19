import pandas as pd

df = pd.read_csv('table.csv')
# Count number of conferences with a non-"-" value in the '1996' column
count_1996 = df['IEEE Conference on Computational Complexity'].notna().sum()
# Actually, we need to check the specific column at index 3 (1996)
# Correct approach: access the column at index 3 (1996)
column_1996 = df.iloc[:, 3]  # 1996 column
count_1996 = (column_1996 != '-').sum()
print(f"Final Answer: {count_1996}")