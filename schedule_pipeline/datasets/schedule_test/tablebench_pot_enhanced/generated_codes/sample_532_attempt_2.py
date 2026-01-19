import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where Confluence [by Lahn-km] is less than 100 and not null
valid_rivers = df[df['Confluence [by Lahn-km]'] < 100]
# Convert Length [km] to numeric and calculate mean
average_length = valid_rivers['Length [km]'].astype(float).mean()
print(f"Final Answer: {average_length:.1f}")