import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where urban percentage is greater than 50
urban_above_50 = df[df['urban , %'] > 50]
# Get the first year when urbanization surpassed 50%
first_year = urban_above_50['year (january)'].iloc[0]
print(f"Final Answer: {first_year}")