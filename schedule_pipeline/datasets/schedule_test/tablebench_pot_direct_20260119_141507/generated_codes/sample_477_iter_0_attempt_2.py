import pandas as pd

df = pd.read_csv('table.csv')
# Calculate mean and standard deviation of points
mean_points = df['points'].mean()
std_points = df['points'].std()

# Identify outliers using 2 standard deviations rule
outliers = df[(df['points'] > mean_points + 2 * std_points) | (df['points'] < mean_points - 2 * std_points)]

# Extract the song names of outliers
outlier_songs = outliers['song'].tolist()

print(f"Final Answer: {', '.join(outlier_songs)}")