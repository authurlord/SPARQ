import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('table.csv')

# Clean the data: remove rows with invalid or non-numeric entries
# First, filter out rows where 'Region' is not a valid region (skip Greece, EU, etc.)
df_clean = df[df['Region'].str.strip() != 'Greece']
df_clean = df_clean[df_clean['Region'].str.strip() != 'European Union']
df_clean = df_clean[df_clean['Region'].str.strip() != '']

# Remove rows where any of the required columns contain non-numeric values
df_clean = df_clean.dropna(subset=['GDP (€, billions)', 'GDP per capita (€)', 'GDP (% of national total)'])

# Convert columns to numeric
df_clean['GDP (€, billions)'] = pd.to_numeric(df_clean['GDP (€, billions)'], errors='coerce')
df_clean['GDP per capita (€)'] = pd.to_numeric(df_clean['GDP per capita (€)'], errors='coerce')
df_clean['GDP (% of national total)'] = pd.to_numeric(df_clean['GDP (% of national total)'], errors='coerce')

# Filter regions where GDP (% of national total) > 5%
filtered_df = df_clean[df_clean['GDP (% of national total)'] > 5]

# If no data remains, return a message
if filtered_df.empty:
    print("Final Answer: No data available for regions with GDP (% of national total) > 5%")
else:
    # Compute the correlation between GDP (€, billions) and GDP per capita (€)
    correlation = filtered_df['GDP (€, billions)'].corr(filtered_df['GDP per capita (€)'])
    print(f"Final Answer: {correlation:.2f}")