import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for year 2004 and sum the mintage
total_mintage_2004 = df[df['year'] == '2004']['mintage'].sum()
print(f"Final Answer: {total_mintage_2004}")