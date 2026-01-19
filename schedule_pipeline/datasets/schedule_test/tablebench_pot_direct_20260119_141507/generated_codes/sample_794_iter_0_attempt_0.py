import pandas as pd

df = pd.read_csv('table.csv')

# Extract the 'troops per one million population' column
troops_per_million = df['troops per one million population'].dropna()

# Get the top two values
top_two = troops_per_million.nlargest(2)
highest = top_two.iloc[0]
second_highest = top_two.iloc[1]

# Calculate percentage difference
percentage_difference = ((highest - second_highest) / second_highest) * 100

# Get corresponding country names
country_highest = df.loc[df['troops per one million population'] == highest, 'country'].values[0]
country_second_highest = df.loc[df['troops per one million population'] == second_highest, 'country'].values[0]

print(f"Final Answer: {country_highest}, {percentage_difference:.2f}%")