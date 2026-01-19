import pandas as pd

# Load the table
df = pd.read_csv('table.csv')

# Clean and filter data
# Remove rows with invalid GDP (% of national total) or missing values
df_clean = df[df['GDP (% of national total)'].str.strip().astype(float).notna()]

# Filter only regions where GDP (% of national total) > 5
df_filtered = df_clean[df_clean['GDP (% of national total)'] > 5]

# Convert GDP (€, billions) and GDP per capita (€) to numeric
df_filtered['GDP (€, billions)'] = pd.to_numeric(df_filtered['GDP (€, billions)'], errors='coerce')
df_filtered['GDP per capita (€)'] = pd.to_numeric(df_filtered['GDP per capita (€)'], errors='coerce')

# Drop rows with NaN after conversion
df_filtered = df_filtered.dropna(subset=['GDP (€, billions)', 'GDP per capita (€)'])

# Compute the correlation between GDP (€, billions) and GDP per capita (€)
correlation = df_filtered['GDP (€, billions)'].corr(df_filtered['GDP per capita (€)'])

print(f"Final Answer: {correlation:.2f}")