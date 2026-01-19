import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Basic insights:
# 1. Maximum and minimum speakers
max_speakers = df['speakers'].max()
min_speakers = df['speakers'].min()

# 2. Average number of speakers
avg_speakers = df['speakers'].mean()

# 3. Top council area by speakers
top_area = df.loc[df['speakers'].idxmax(), 'council area']

# 4. Bottom council area by speakers
bottom_area = df.loc[df['speakers'].idxmin(), 'council area']

# 5. Correlation between speakers and population (optional insight)
# We can check if there's a trend, but since it's not asked directly, we'll just summarize

print(f"Final Answer: {max_speakers}, {min_speakers}, {avg_speakers:.0f}, {top_area}, {bottom_area}")