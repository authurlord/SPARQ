import pandas as pd

df = pd.read_csv('table.csv')
# Convert Year to integer, then filter for Glen Campbell songs in the 1980s
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
glen_campbell_1980s = df[(df['Artist'] == 'Glen Campbell') & (df['Year'] >= 1980) & (df['Year'] <= 1989)]
count = len(glen_campbell_1980s)
print(f"Final Answer: {count}")