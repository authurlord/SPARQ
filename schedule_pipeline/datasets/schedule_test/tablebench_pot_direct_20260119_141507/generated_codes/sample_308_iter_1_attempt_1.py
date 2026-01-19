import pandas as pd

df = pd.read_csv('table.csv')
# Filter songs by Glen Campbell and year in the 1980s
glen_campbell_1980s = df[(df['Artist'] == 'Glen Campbell') & (df['Year'].astype(int) >= 1980) & (df['Year'].astype(int) <= 1989)]
count = len(glen_campbell_1980s)
print(f"Final Answer: {count}")