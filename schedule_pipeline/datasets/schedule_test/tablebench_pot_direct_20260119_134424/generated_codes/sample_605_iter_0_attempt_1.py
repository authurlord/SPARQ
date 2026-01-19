import pandas as pd

df = pd.read_csv('table.csv')
# Filter the data for the year 2005
year_2005 = df[df['year'] == '2005']
# Sum the number of Indians and Pakistanis admitted
total_admitted = year_2005['indians admitted'].astype(int).sum() + year_2005['pakistanis admitted'].astype(int).sum()
print(f"Final Answer: {total_admitted}")