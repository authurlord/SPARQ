import pandas as pd

df = pd.read_csv('table.csv')
# Filter for 7th round
seventh_round = df[df['round'] == '7']
# Calculate Draft Position Improvement
seventh_round['draft_improvement'] = seventh_round['overall'].astype(int) - seventh_round['pick'].astype(int)
# Extract the improvement values
improvements = seventh_round['draft_improvement'].tolist()
print(f"Final Answer: {', '.join(map(str, improvements))}")