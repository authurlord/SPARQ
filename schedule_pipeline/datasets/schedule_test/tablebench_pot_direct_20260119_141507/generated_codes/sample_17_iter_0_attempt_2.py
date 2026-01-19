import pandas as pd

df = pd.read_csv('table.csv')
# Sum the 'total' column to get the total number of medals
total_medals = df['total'].sum()
print(f"Final Answer: {total_medals}")