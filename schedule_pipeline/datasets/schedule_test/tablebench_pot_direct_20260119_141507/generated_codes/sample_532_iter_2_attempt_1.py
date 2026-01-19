import pandas as pd

df = pd.read_csv('table.csv')

# Filter rivers with confluence < 100 km and not null
filtered_df = df[(df['Confluence [by Lahn-km]'].notna()) & (df['Confluence [by Lahn-km]'] < 100)]

# Calculate average length of such rivers
average_length = filtered_df['Length [km]'].mean()

print(f"Final Answer: {average_length:.1f}")