import pandas as pd

df = pd.read_csv('table.csv')
# Filter players from the 7th round
seventh_round = df[df['round'] == '7']
# Calculate draft position improvement: overall - pick
seventh_round['improvement'] = seventh_round['overall'].astype(int) - seventh_round['pick'].astype(int)
# Calculate average improvement
avg_improvement = seventh_round['improvement'].mean()
print(f"Final Answer: {avg_improvement:.1f}")