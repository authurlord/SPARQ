import pandas as pd

df = pd.read_csv('table.csv')
# Filter players drafted in the 7th round
round_7_players = df[df['round'] == '7']
# Convert pick and overall to integers
improvement = round_7_players['overall'].astype(int) - round_7_players['pick'].astype(int)
print(f"Final Answer: {improvement.tolist()}")