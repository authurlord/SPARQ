import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert 'Confluence [by Lahn-km]' to numeric, replacing NaN with None
df['Confluence [by Lahn-km]'] = pd.to_numeric(df['Confluence [by Lahn-km]'], errors='coerce')

# Filter rows where confluence is below 100 km and not null
filtered_df = df[df['Confluence [by Lahn-km]'] < 100]

# Calculate the average length of these rivers
average_length = filtered_df['Length [km]'].mean()

print(f"Final Answer: {average_length:.1f}")