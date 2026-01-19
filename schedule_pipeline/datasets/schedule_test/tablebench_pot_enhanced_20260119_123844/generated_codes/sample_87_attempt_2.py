import pandas as pd

df = pd.read_csv('table.csv')
# Filter the player with average yards of 11.4
player = df[df['Avg.'] == '11.4']['Player'].iloc[0]
print(f"Final Answer: {player}")