import pandas as pd

df = pd.read_csv('table.csv')
# Extract the 'spanish' row and sum across municipalities
spanish_population = df[df['language'] == 'spanish'].iloc[0, 1:]
municipality_with_highest_spanish = spanish_population.idxmax()
print(f"Final Answer: {municipality_with_highest_spanish}")