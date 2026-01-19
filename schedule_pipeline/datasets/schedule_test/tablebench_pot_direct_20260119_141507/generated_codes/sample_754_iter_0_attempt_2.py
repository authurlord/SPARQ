import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Convert 'year' to integer and ensure proper order
df['year'] = pd.to_numeric(df['year'], errors='coerce')

# Sort by year to ensure chronological order
df = df.sort_values('year').reset_index(drop=True)

# Calculate the trend in qualifying speed
qual_trend = df['qual'].values

# Check if the trend is increasing or decreasing
trend_direction = "increasing" if df['qual'].iloc[-1] > df['qual'].iloc[0] else "decreasing"

# Print the trend and a brief explanation
print(f"Final Answer: {trend_direction}")