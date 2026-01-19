import pandas as pd

df = pd.read_csv('table.csv')
# Select the first 4 years (2002-2006)
first_four_years = df.head(4)
# Sum the mintage values
total_mintage = first_four_years['mintage'].sum()
print(f"Final Answer: {total_mintage}")