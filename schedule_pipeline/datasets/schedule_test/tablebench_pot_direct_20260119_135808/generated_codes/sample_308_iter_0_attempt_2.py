import pandas as pd

df = pd.read_csv('table.csv')
# Filter for Glen Campbell and songs released in the 1980s
glen_campbell_1980s = df[(df['Artist'] == 'Glen Campbell') & (df['Year'].astype(int).between(1980, 1989))]
count = len(glen_campbell_1980s)
print(f"Final Answer: {count}")