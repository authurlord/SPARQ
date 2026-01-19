import pandas as pd

df = pd.read_csv('table.csv')
# Filter the row for Lionel Palairet
lionel_palairet = df[df['player'] == 'lionel palairet']
# Extract the highest score
highest_score = lionel_palairet['highest score'].values[0]
print(f"Final Answer: {highest_score}")