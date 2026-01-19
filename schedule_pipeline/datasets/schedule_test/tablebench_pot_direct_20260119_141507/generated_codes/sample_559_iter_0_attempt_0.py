import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')
# Filter rows where year is between 1895 and 1980 inclusive
filtered_df = df[(df['year'] >= 1895) & (df['year'] <= 1980)]
# Remove rows where 'floors' is missing (i.e., '-')
filtered_df['floors'] = filtered_df['floors'].replace('-', np.nan)
# Drop rows with NaN in 'floors'
filtered_df = filtered_df.dropna(subset=['floors'])
# Calculate the mean of floors
avg_floors = filtered_df['floors'].mean()
print(f"Final Answer: {avg_floors:.1f}")