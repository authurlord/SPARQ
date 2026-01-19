import pandas as pd

df = pd.read_csv('table.csv')

# Identify the surname with the highest number of bearers in 1971
max_1971 = df.loc[df['number of bearers 1971'].idxmax(), 'surname']

# Identify the surname with the lowest number of bearers in 2009
min_2009 = df.loc[df['number of bearers 2009'].idxmin(), 'surname']

print(f"Final Answer: {max_1971}, {min_2009}")