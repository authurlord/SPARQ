import pandas as pd

df = pd.read_csv('table.csv')
# Find the row where language is 'spanish'
spanish_row = df[df['language'] == 'spanish']
# Get the municipality with the highest value in that row
max_population_municipality = spanish_row.idxmax(axis=1)[0]
# Since idxmax returns index, we get the column name by using the column name directly
# Actually, we want the column name (municipality) with max value
max_value = spanish_row.iloc[0].max()
# Get the column name of the max value
municipality_with_max_spanish = spanish_row.columns[spanish_row.iloc[0].idxmax()]
print(f"Final Answer: {municipality_with_max_spanish}")