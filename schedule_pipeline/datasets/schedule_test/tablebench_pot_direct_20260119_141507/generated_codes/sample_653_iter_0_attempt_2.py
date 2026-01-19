import pandas as pd

df = pd.read_csv('table.csv')
# Calculate the correlation between 'Length [km]' and 'Drainage basin area [km2]'
correlation = df['Length [km]'].corr(df['Drainage basin area [km2]'])
print(f"Final Answer: {correlation:.3f}")