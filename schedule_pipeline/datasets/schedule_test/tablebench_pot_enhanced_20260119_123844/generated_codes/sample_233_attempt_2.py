import pandas as pd

df = pd.read_csv('table.csv')

# Filter rows where GDP (% of national total) > 5
filtered_df = df[df['GDP (% of national total)'] != '-']
filtered_df['GDP (% of national total)'] = pd.to_numeric(filtered_df['GDP (% of national total)'])
filtered_df = filtered_df[filtered_df['GDP (% of national total)'] > 5]

# Extract GDP (€, billions) and GDP per capita (€)
gdp_billion = filtered_df['GDP (€, billions)'].astype(float)
gdp_per_capita = filtered_df['GDP per capita (€)'].astype(str).str.replace(',', '').astype(float)

# Display the relationship
print("GDP (€, billions) vs GDP per capita (€) for regions with GDP > 5% of national total:")
for i in range(len(filtered_df)):
    print(f"{filtered_df.iloc[i]['Region']}: {gdp_billion.iloc[i]:.3f} billion € → {gdp_per_capita.iloc[i]:.0f} €")

# Optional: Show the trend via a simple print of the correlation
correlation = gdp_billion.corr(gdp_per_capita)
print(f"Correlation between GDP and GDP per capita: {correlation:.2f}")

# Final Answer: List of regions with GDP > 5% and their corresponding GDP per capita (€)
regions = filtered_df['Region'].tolist()
gdp_per_capita_values = gdp_per_capita.tolist()
result = [f"{r} ({v})" for r, v in zip(regions, gdp_per_capita_values)]
print(f"Final Answer: {', '.join(result)}")