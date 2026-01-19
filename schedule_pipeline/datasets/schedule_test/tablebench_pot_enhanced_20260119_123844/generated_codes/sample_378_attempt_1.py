import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where seats >= 75
winning_elections = df[df['seats'] >= 75]
# Count the number of such elections
count = len(winning_elections)
print(f"Final Answer: {count}")