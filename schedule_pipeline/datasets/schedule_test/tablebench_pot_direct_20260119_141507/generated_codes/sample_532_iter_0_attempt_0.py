import pandas as pd

df = pd.read_csv('table.csv')
# Filter rivers where confluence is less than 100 km and not null
filtered_rivers = df[df['Confluence [by Lahn-km]'].notna() & (df['Confluence [by Lahn-km]'] < 100)]
# Calculate the average length of these rivers
average_length = filtered_rivers['Length [km]'].mean()
print(f"Final Answer: {average_length:.1f}")