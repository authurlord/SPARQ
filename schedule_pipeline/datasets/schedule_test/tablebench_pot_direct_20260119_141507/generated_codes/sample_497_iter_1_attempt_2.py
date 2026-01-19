import pandas as pd

df = pd.read_csv('table.csv')
# Filter players drafted in the 7th round
round_7_players = df[df['round'] == '7']
# Compute improvement: overall - pick
improvement = round_7_players['overall'] - round_7_players['pick']
# Convert to list and print
print(f"Final Answer: {improvement.tolist()}")