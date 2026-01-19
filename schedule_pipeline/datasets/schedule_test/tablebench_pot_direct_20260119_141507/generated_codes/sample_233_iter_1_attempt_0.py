import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Convert necessary columns to numeric, handling errors by converting invalid entries to NaN
df['GDP (€, billions)'] = pd.to_numeric(df['GDP (€, billions)'], errors='coerce')
df['GDP per capita (€)'] = pd.to_numeric(df['GDP per capita (€)'], errors='coerce')
df['GDP (% of national total)'] = pd.to_numeric(df['GDP (% of national total)'], errors='coerce')

# Filter regions where GDP (% of national total) > 5
filtered_df = df[(df['GDP (% of national total)'] > 5) & (df['GDP (€, billions)'].notna()) & (df['GDP per capita (€)'].notna())]

# Drop rows where any value is missing after filtering
filtered_df = filtered_df.dropna()

# Calculate the correlation between GDP (€, billions) and GDP per capita (€)
correlation = filtered_df['GDP (€, billions)'].corr(filtered_df['GDP per capita (€)'])

print(f"Final Answer: {correlation:.3f}")