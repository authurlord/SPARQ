import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where 'Confluence [by Lahn-km]' is less than 100 and not null
filtered_df = df[df['Confluence [by Lahn-km]'].notna() & (df['Confluence [by Lahn-km]'] < 100)]

# Calculate the average length of these rivers
average_length = filtered_df['Length [km]'].mean()
print(f"Final Answer: {average_length:.1f}")