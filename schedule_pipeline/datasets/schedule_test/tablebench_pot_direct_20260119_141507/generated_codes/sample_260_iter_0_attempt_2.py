import pandas as pd
import re

df = pd.read_csv('table.csv')

# Extract year from "Date Built" and "Date Withdrawn"
df['built_year'] = df['Date Built'].str.extract(r'(\d{4})')[0].astype(int)
df['withdrawn_year'] = df['Date Withdrawn'].str.extract(r'(\d{4})')[0].astype(int)

# Filter locomotives built in 1938
filtered_df = df[df['built_year'] == 1938]

# Find the maximum service duration (latest withdrawal year - 1938)
max_service_years = filtered_df['withdrawn_year'].max() - 1938

print(f"Final Answer: {max_service_years}")