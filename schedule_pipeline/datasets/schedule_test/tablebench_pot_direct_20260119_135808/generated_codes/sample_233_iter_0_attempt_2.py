import pandas as pd

df = pd.read_csv('table.csv')

# Filter rows where GDP (% of national total) > 5
filtered_df = df[df['GDP (% of national total)'] != '–']
filtered_df['GDP (% of national total)'] = pd.to_numeric(filtered_df['GDP (% of national total)'])
filtered_df = filtered_df[filtered_df['GDP (% of national total)'] > 5]

# Select relevant columns and sort by GDP (€, billions) for clarity
result = filtered_df[['Region', 'GDP (€, billions)', 'GDP per capita (€)']].sort_values(by='GDP (€, billions)', ascending=False)

print(result)