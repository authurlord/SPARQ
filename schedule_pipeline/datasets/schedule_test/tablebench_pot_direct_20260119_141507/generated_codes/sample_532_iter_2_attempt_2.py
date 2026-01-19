import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert confluence column to numeric, coercing errors to NaN
df['Confluence [by Lahn-km]'] = pd.to_numeric(df['Confluence [by Lahn-km]'], errors='coerce')

# Filter rivers with confluence below 100 km
filtered_rivers = df[df['Confluence [by Lahn-km]'] < 100]

# Calculate average length of such rivers
if not filtered_rivers.empty:
    avg_length = filtered_rivers['Length [km]'].mean()
    print(f"Final Answer: {avg_length:.1f}")
else:
    print("Final Answer: 0.0")