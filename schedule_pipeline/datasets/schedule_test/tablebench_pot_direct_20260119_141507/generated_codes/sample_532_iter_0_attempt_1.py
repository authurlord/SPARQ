import pandas as pd

df = pd.read_csv('table.csv')
# Filter rivers where confluence is below 100 km
filtered_rivers = df[df['Confluence [by Lahn-km]'].fillna(0).astype(float) < 100]
# Calculate the average length of these rivers
average_length = filtered_rivers['Length [km]'].mean()
print(f"Final Answer: {average_length:.1f}")