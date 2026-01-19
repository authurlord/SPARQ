import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Filter for years 2000 to 2004 (first 5 years of 2000s)
filtered_df = df[df['year'].isin(['2000', '2001', '2002', '2004'])]

# Convert 'mintage (proof)' to numeric, handling errors by converting invalid entries to NaN
filtered_df['mintage (proof)'] = pd.to_numeric(filtered_df['mintage (proof)'], errors='coerce')

# Calculate average of valid mintage (proof) values
average_mintage = filtered_df['mintage (proof)'].mean()

print(f"Final Answer: {average_mintage:.0f}")