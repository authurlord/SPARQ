import pandas as pd

df = pd.read_csv('table.csv')
# Find the row where country is 'texas' and get the 'average' value
texas_average = df[df['country'] == 'texas']['average'].values[0]
print(f"Final Answer: {texas_average}")