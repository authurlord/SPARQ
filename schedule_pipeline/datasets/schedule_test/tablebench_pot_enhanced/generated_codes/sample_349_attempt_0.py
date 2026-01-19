import pandas as pd

df = pd.read_csv('table.csv')
# Count non-null entries in the '1996' column
count_1996 = df['1996'].notna().sum()
print(f"Final Answer: {count_1996}")