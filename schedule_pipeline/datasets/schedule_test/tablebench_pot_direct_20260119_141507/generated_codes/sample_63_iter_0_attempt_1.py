import pandas as pd

df = pd.read_csv('table.csv')
# Find the highest score for Lionel Palairet
highest_score = df[df['player'] == 'lionel palairet']['highest score'].values[0]
print(f"Final Answer: {highest_score}")