import pandas as pd

df = pd.read_csv('table.csv')
# Filter the row for Texas
texas_average = df[df['country'] == 'texas']['average'].values[0]
print(f"Final Answer: {texas_average}")