import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Convert 'year' to integer and sort by year
df['year'] = df['year'].astype(int)
df = df.sort_values('year')

# Extract 'total s ton' column
total_s_ton = df['total s ton'].values

# Compute year-over-year increases
increases = []
for i in range(1, len(total_s_ton)):
    increase = total_s_ton[i] - total_s_ton[i-1]
    increases.append(increase)

# Find the index of the maximum increase and get the corresponding year
max_increase_index = increases.index(max(increases)) + 1  # +1 because it starts at 2001
max_year = df['year'].iloc[max_increase_index]

print(f"Final Answer: {max_year}")