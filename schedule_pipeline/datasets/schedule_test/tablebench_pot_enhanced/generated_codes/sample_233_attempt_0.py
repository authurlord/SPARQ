import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Filter regions where GDP (% of national total) > 5%
filtered_df = df[df['GDP (% of national total)'] != '0']
filtered_df = filtered_df[filtered_df['GDP (% of national total)'].astype(float) > 5]

# Convert GDP (€, billions) and GDP per capita (€) to numeric
filtered_df['GDP (€, billions)'] = pd.to_numeric(filtered_df['GDP (€, billions)'])
filtered_df['GDP per capita (€)'] = pd.to_numeric(filtered_df['GDP per capita (€)'].str.replace(',', ''))

# Plot the relationship
plt.figure(figsize=(10, 6))
plt.scatter(filtered_df['GDP (€, billions)'], filtered_df['GDP per capita (€)'], color='blue')
plt.title('GDP per capita (€) vs GDP (€, billions) for Regions with GDP > 5% of National Total')
plt.xlabel('GDP (€, billions)')
plt.ylabel('GDP per capita (€)')
plt.grid(True)
plt.show()

# Output the trend observation
print("Final Answer: Increasing GDP (€, billions) is generally associated with higher GDP per capita (€), but with some variation.")