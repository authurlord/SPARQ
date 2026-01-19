import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'troops per one million population' to numeric for comparison
df['troops per one million population'] = pd.to_numeric(df['troops per one million population'], errors='coerce')

# Sort by 'troops per one million population' in descending order
sorted_df = df.sort_values(by='troops per one million population', ascending=False)

# Get the top two countries
highest_country = sorted_df.iloc[0]['country']
second_highest_value = sorted_df.iloc[1]['troops per one million population']
highest_value = sorted_df.iloc[0]['troops per one million population']

# Calculate percentage difference
percentage_diff = ((highest_value - second_highest_value) / second_highest_value) * 100

print(f"Final Answer: {highest_country}, {percentage_diff:.2f}")