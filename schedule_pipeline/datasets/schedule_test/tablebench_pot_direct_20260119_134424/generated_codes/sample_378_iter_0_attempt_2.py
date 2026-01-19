import pandas as pd

df = pd.read_csv('table.csv')
# Count elections with 75 or more seats
count_elections = df[df['seats'] >= 75].shape[0]
print(f"Final Answer: {count_elections}")