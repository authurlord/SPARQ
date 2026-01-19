import pandas as pd

df = pd.read_csv('table.csv')
# Filter players in the 7th round
seventh_round_players = df[df['round'] == '7']
# Convert pick and overall to integers
seventh_round_players['pick'] = pd.to_numeric(seventh_round_players['pick'], errors='coerce')
seventh_round_players['overall'] = pd.to_numeric(seventh_round_players['overall'], errors='coerce')
# Calculate improvement: overall - pick
improvement = seventh_round_players['overall'] - seventh_round_players['pick']
# Print the improvement values
print(f"Final Answer: {improvement.tolist()}")