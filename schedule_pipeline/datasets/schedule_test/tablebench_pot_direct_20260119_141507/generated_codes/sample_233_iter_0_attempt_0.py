import pandas as pd

df = pd.read_csv('table.csv')

# Filter rows where GDP (% of national total) > 5%
filtered_df = df[df['GDP (% of national total)'].str.strip().astype(float) > 5]

# Drop rows with missing or invalid values in GDP (€, billions) or GDP per capita (€)
filtered_df = filtered_df.dropna(subset=['GDP (€, billions)', 'GDP per capita (€)'])

# Convert relevant columns to numeric
filtered_df['GDP (€, billions)'] = pd.to_numeric(filtered_df['GDP (€, billions)'], errors='coerce')
filtered_df['GDP per capita (€)'] = pd.to_numeric(filtered_df['GDP per capita (€)'], errors='coerce')

# Remove any remaining NaNs
filtered_df = filtered_df.dropna()

# Compute correlation between GDP (€, billions) and GDP per capita (€)
correlation = filtered_df['GDP (€, billions)'].corr(filtered_df['GDP per capita (€)'])

print(f"Final Answer: {correlation:.2f}")