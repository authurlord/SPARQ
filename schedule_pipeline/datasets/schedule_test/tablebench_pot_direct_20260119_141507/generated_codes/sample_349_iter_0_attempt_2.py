import pandas as pd

df = pd.read_csv('table.csv')
# Count the number of conferences that have a non-'-' value in the '1996' column
count_1996 = df['1996'].astype(str).str.strip().ne('-').sum()
print(f"Final Answer: {count_1996}")