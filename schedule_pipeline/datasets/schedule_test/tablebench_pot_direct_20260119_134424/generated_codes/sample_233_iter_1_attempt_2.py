import pandas as pd

df = pd.read_csv('table.csv')

# Filter regions with GDP (% of national total) > 5%
filtered_df = df[df['GDP (% of national total)'] != '-']
filtered_df = filtered_df[filtered_df['GDP (% of national total)'].astype(float) > 5.0]

# Convert GDP (€, billions) and GDP per capita (€) to numeric, handling commas
filtered_df['GDP (€, billions)'] = filtered_df['GDP (€, billions)'].str.replace(',', '').astype(float)
filtered_df['GDP per capita (€)'] = filtered_df['GDP per capita (€)'].str.replace(',', '').astype(float)

# Sort by GDP (€, billions) to see the trend
sorted_df = filtered_df.sort_values(by='GDP (€, billions)', ascending=False)

# Extract the relevant data for final answer
result = sorted_df[['Region', 'GDP (€, billions)', 'GDP per capita (€)']]

# Print the result
print(result.to_string(index=False))

# Final Answer: List of regions with their GDP and GDP per capita
# Format: Region, GDP (€, billions), GDP per capita (€)
final_output = ", ".join([f"{row['Region']} ({row['GDP (€, billions)']:.1f}, {row['GDP per capita (€)']})" for _, row in sorted_df.iterrows()])
print(f"Final Answer: {final_output}")