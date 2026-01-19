import pandas as pd

df = pd.read_csv('table.csv')
# Filter the row where SPECIFICATION_1 is 'Males'
males_row = df[df['SPECIFICATION_1'] == 'Males']
# Extract the population values for 20–29 and 30–39 age groups
males_20_29 = males_row['POPULATION (by age group in 2002)_3'].values[0]
males_30_39 = males_row['POPULATION (by age group in 2002)_4'].values[0]
# Calculate the total
total_males = males_20_29 + males_30_39
print(f"Final Answer: {total_males}")