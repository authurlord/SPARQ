import pandas as pd

df = pd.read_csv('table.csv')
# Filter for 7th round
seventh_round = df[df['round'] == '7']
# Calculate draft position improvement
seventh_round['improvement'] = seventh_round['overall'].astype(int) - seventh_round['pick'].astype(int)
# Extract improvement values
improvements = seventh_round['improvement'].tolist()
print(f"Final Answer: {', '.join(map(str, improvements))}")