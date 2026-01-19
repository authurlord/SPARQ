import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert confluence to numeric, coercing errors to NaN
df['Confluence [by Lahn-km]'] = pd.to_numeric(df['Confluence [by Lahn-km']], errors='coerce')

# Filter rivers where confluence is below 100 km and not NaN
filtered_rivers = df[df['Confluence [by Lahn-km]'] < 100]

# Calculate average length of these rivers
average_length = filtered_rivers['Length [km]'].mean()

print(f"Final Answer: {average_length:.1f}")