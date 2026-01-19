import pandas as pd

df = pd.read_csv('table.csv')
# Filter songs by Glen Campbell released in the 1980s
glen_1980s = df[(df['Artist'] == 'Glen Campbell') & (df['Year'].between(1980, 1989))]
count = len(glen_1980s)
print(f"Final Answer: {count}")