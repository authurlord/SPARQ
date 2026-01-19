import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Ensure 'speakers' column is of integer type
df['speakers'] = pd.to_numeric(df['speakers'], errors='coerce')

# Drop any rows with invalid speaker counts (if any)
df = df.dropna(subset=['speakers'])

# Describe the distribution of speakers
print(f"Total council areas: {len(df)}")
print(f"Maximum speakers: {df['speakers'].max()}")
print(f"Minimum speakers: {df['speakers'].min()}")
print(f"Average speakers: {df['speakers'].mean():.1f}")
print(f"Median speakers: {df['speakers'].median()}")

# Identify the council area with the most and least speakers
max_speakers_area = df.loc[df['speakers'].idxmax(), 'council area']
min_speakers_area = df.loc[df['speakers'].idxmin(), 'council area']

print(f"Area with most speakers: {max_speakers_area}")
print(f"Area with least speakers: {min_speakers_area}")

Final Answer: 15811, 97