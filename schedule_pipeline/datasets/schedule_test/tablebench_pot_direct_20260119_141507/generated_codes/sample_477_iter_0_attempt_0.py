import pandas as pd

df = pd.read_csv('table.csv')
# Calculate mean and std of points
mean_points = df['points'].mean()
std_points = df['points'].std()

# Identify outliers using 2 standard deviations rule
outlier_threshold = 2 * std_points
outliers = df[(df['points'] > mean_points + outlier_threshold) | (df['points'] < mean_points - outlier_threshold)]

# Extract the song names of outliers
outlier_songs = outliers['song'].tolist()

# Also check for negative points (which may indicate an anomaly)
negative_points = df[df['points'] < 0]['song'].tolist()

# Combine both types of anomalies
all_outlier_songs = list(set(outlier_songs + negative_points))

print(f"Final Answer: {', '.join(all_outlier_songs)}")