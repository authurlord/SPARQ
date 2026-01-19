import pandas as pd

df = pd.read_csv('table.csv')
# Find the row where player is 'Lionel Palairet' and get the highest score
highest_score = df[df['player'] == 'lionel palairet']['highest score'].values[0]
print(f"Final Answer: {highest_score}")