import pandas as pd

df = pd.read_csv('table.csv')
# Extract population of 6th ranked city ('valenza') and 'acqui terme'
pop_valenza = df[df['city'] == 'valenza']['population'].values[0]
pop_acqui_terme = df[df['city'] == 'acqui terme']['population'].values[0]
difference = abs(pop_valenza - pop_acqui_terme)
print(f"Final Answer: {difference}")