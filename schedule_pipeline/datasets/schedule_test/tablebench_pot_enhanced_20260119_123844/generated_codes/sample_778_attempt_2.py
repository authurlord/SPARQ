import pandas as pd

df = pd.read_csv('table.csv')

# Extract the 'Year_2' and corresponding values (last column: 'Year_2')
years = df['Year_2'].astype(int)
values = df['-_2'].str.replace(',', '').astype(int)

# Calculate the average annual increase from 2005 to 2010
start_idx = df[df['Year_2'] == '2005'].index[0]
end_idx = df[df['Year_2'] == '2010'].index[0]

annual_increase = (values[end_idx] - values[start_idx]) / (years[end_idx] - years[start_idx])

# Project from 2010 to 2020 (10 years)
projected_value = values[end_idx] + (annual_increase * 10)

print(f"Final Answer: {int(projected_value)}")