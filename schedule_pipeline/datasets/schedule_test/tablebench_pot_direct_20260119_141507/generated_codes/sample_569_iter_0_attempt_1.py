import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('table.csv')

# Replace "no data" with NaN and convert to numeric
df['CPSC toy safety funding (US$ Millions)'] = df['CPSC toy safety funding (US$ Millions)'].replace('no data', np.nan).astype(float)
df['Injuries (US $000)'] = df['Injuries (US $000)'].replace('no data', np.nan).astype(float)

# Remove rows where CPSC funding is missing
df_clean = df.dropna(subset=['CPSC toy safety funding (US$ Millions)'])

# Calculate average funding per year
df_clean['avg_funding'] = df_clean['CPSC toy safety funding (US$ Millions)']

# Sort by average funding in descending order and take top 3 years
top_3_years = df_clean.nlargest(3, 'CPSC toy safety funding (US$ Millions)')['Year']

# Filter the original data to get injuries for those years
injury_sum = df[df['Year'].isin(top_3_years)]['Injuries (US $000)'].sum()

print(f"Final Answer: {injury_sum}")