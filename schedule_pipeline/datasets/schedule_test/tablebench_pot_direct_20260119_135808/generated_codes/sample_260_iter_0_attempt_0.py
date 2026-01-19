import pandas as pd
from datetime import datetime

df = pd.read_csv('table.csv')

# Convert 'Date Built' and 'Date Withdrawn' to datetime objects
df['Date Built'] = pd.to_datetime(df['Date Built'], format='%B %Y')
df['Date Withdrawn'] = pd.to_datetime(df['Date Withdrawn'], format='%B %Y')

# Filter for locomotives built in 1938
df_1938 = df[df['Date Built'].dt.year == 1938]

# Calculate the number of years in service
df_1938['Years in Service'] = (df_1938['Date Withdrawn'] - df_1938['Date Built']).dt.days / 365.25

# Find the maximum number of years
max_years = df_1938['Years in Service'].max()

# Round to nearest whole number
max_years_rounded = round(max_years)

print(f"Final Answer: {max_years_rounded}")