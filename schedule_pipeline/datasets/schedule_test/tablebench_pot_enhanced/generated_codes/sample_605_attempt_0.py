import pandas as pd

df = pd.read_csv('table.csv')
# Filter the data for the year 2005
year_2005 = df[df['year'] == '2005']
# Sum the number of Indians and Pakistanis admitted
total_admitted_2005 = int(year_2005['indians admitted'].values[0]) + int(year_2005['pakistanis admitted'].values[0])
print(f"Final Answer: {total_admitted_2005}")