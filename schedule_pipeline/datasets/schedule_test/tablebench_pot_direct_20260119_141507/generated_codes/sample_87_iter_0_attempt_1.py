import pandas as pd

df = pd.read_csv('table.csv')
# Find the player with an average of 11.4 yards per reception
player = df[df['Avg.'] == '11.4']['Player'].values[0]
print(f"Final Answer: {player}")