import pandas as pd

df = pd.read_csv('table.csv')

# Filter rows for 'north' and 'liaoning'
north_manchu = df[df['region'] == 'north']['manchu'].values[0]
liaoning_manchu = df[df['region'] == 'liaoning']['manchu'].values[0]

# Convert to integers (they are strings in the data)
north_manchu = int(north_manchu)
liaoning_manchu = int(liaoning_manchu)

# Calculate the minimum increase needed
increase_needed = liaoning_manchu - north_manchu + 1  # +1 to surpass

# Calculate percentage increase
percentage_increase = (increase_needed / north_manchu) * 100

print(f"Final Answer: {percentage_increase:.2f}")