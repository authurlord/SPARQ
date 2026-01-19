import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where confluence is below 100 km and not null
filtered_df = df[df['Confluence [by Lahn-km]'].notna() & (df['Confluence [by Lahn-km]'] < 100)]
# Convert Length [km] to float and calculate average
average_length = filtered_df['Length [km]'].astype(float).mean()
print(f"Final Answer: {average_length:.1f}")