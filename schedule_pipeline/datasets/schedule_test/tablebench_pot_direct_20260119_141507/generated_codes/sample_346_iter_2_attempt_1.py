import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where 'spanish' is 40,000 or more and count them
count_spanish_40k = (df['spanish'] >= 40000).sum()
print(f"Final Answer: {count_spanish_40k}")