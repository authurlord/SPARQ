import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Convert 'speakers' column to integer (previously failed due to string parsing)
df['speakers'] = df['speakers'].str.replace(',', '').astype(int)

# Basic insights
max_speakers = df['speakers'].max()
min_speakers = df['speakers'].min()
median_speakers = df['speakers'].median()
area_with_most_speakers = df.loc[df['speakers'].idxmax(), 'council area']
area_with_least_speakers = df.loc[df['speakers'].idxmin(), 'council area']

# Print insights
print(f"Final Answer: The council area with the most speakers is {area_with_most_speakers} ({max_speakers}), and the one with the least is {area_with_least_speakers} ({min_speakers}). Median speakers: {median_speakers}.")