import pandas as pd

df = pd.read_csv('table.csv')
# Count the number of conferences that have an entry in 1996 (i.e., not '-')
count_1996 = df['1996'].ne('-').sum()
print(f"Final Answer: {count_1996}")