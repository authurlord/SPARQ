import pandas as pd

df = pd.read_csv('table.csv')
# Filter for 7th round
seventh_round = df[df['round'] == '7']
# Calculate draft position improvement
seventh_round['draft_improvement'] = seventh_round['overall'].astype(int) - seventh_round['pick'].astype(int)
# Extract the improvements
improvements = seventh_round['draft_improvement'].tolist()
print(f"Final Answer: {improvements[0]}, {improvements[1]}")