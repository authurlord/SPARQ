import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'troops per one million population' to float for comparison
df['troops per one million population'] = pd.to_numeric(df['troops per one million population'])

# Sort by 'troops per one million population' in descending order
df_sorted = df.sort_values(by='troops per one million population', ascending=False)

# Get the top two countries
highest_country = df_sorted.iloc[0]['country']
second_highest = df_sorted.iloc[1]['troops per one million population']
highest_value = df_sorted.iloc[0]['troops per one million population']

# Calculate percentage difference
percentage_diff = ((highest_value - second_highest) / second_highest) * 100

print(f"Final Answer: {highest_country}, {percentage_diff:.2f}")