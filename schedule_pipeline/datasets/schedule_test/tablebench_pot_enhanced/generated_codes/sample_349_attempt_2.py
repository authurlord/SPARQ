import pandas as pd

df = pd.read_csv('table.csv')
# Count non '-' values in the '1996' column
count_1996 = df['1996'].ne('-').sum()
print(f"Final Answer: {count_1996}")