import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for the first 4 years (2002 to 2006)
first_four_years = df[df['year'].isin(['2002', '2004', '2005', '2006'])]
# Sum the mintage values
total_mintage = first_four_years['mintage'].sum()
print(f"Final Answer: {total_mintage}")