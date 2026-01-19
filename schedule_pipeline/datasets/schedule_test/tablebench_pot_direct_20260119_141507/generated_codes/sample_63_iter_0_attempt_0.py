import pandas as pd

df = pd.read_csv('table.csv')
# Find Lionel Palairet's highest score
highest_score = df[df['player'] == 'lionel palairet']['highest score'].values[0]
print(f"Final Answer: {highest_score}")