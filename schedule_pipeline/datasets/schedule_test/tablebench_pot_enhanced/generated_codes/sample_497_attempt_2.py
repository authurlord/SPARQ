import pandas as pd

df = pd.read_csv('table.csv')
# Filter players drafted in the 7th round
seventh_round_players = df[df['round'] == '7']
# Calculate draft position improvement: overall - pick
seventh_round_players['draft_improvement'] = seventh_round_players['overall'].astype(int) - seventh_round_players['pick'].astype(int)
# Extract the improvements
improvements = seventh_round_players['draft_improvement'].tolist()
print(f"Final Answer: {improvements[0]}, {improvements[1]}")