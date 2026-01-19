import pandas as pd

df = pd.read_csv('table.csv')
# Filter regions with GDP (% of national total) > 5%
filtered_df = df[df['GDP (% of national total)'] > '5.0']
# Convert GDP (€, billions) and GDP per capita (€) to numeric for analysis
filtered_df['GDP (€, billions)'] = pd.to_numeric(filtered_df['GDP (€, billions)'], errors='coerce')
filtered_df['GDP per capita (€)'] = pd.to_numeric(filtered_df['GDP per capita (€)'].str.replace(',', ''), errors='coerce')

# Drop rows with NaN values
filtered_df.dropna(inplace=True)

# Display the filtered data to observe the trend
print("Filtered data (GDP > 5% of national total):")
print(filtered_df[['Region', 'GDP (€, billions)', 'GDP per capita (€)']])

# Calculate correlation coefficient
correlation = filtered_df['GDP (€, billions)'].corr(filtered_df['GDP per capita (€)'])
print(f"Correlation between GDP and GDP per capita: {correlation:.2f}")