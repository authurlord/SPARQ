import pandas as pd

df = pd.read_csv('table.csv')
# Find the nation with total medals equal to 57
nation_57 = df[df['Total'] == 57]['Nation'].values[0]
print(f"Final Answer: {nation_57}")