import pandas as pd

df = pd.read_csv('table.csv')

# Display basic summary of speakers
print(f"Maximum speakers: {df['speakers'].max()} in {df.loc[df['speakers'].idxmax(), 'council area']}")
print(f"Minimum speakers: {df['speakers'].min()} in {df.loc[df['speakers'].idxmin(), 'council area']}")

# Calculate average speakers per population to see density
df['speakers_per_population'] = df['speakers'] / df['population']

# Find the area with highest and lowest speaker-to-population ratio
max_density = df.loc[df['speakers_per_population'].idxmax(), 'council area']
min_density = df.loc[df['speakers_per_population'].idxmin(), 'council area']

print(f"Area with highest speaker-to-population ratio: {max_density}")
print(f"Area with lowest speaker-to-population ratio: {min_density}")

# Final answer: describe key insights based on the data
Final Answer: na h - eileanan siar, highland, city of glasgow, city of edinburgh