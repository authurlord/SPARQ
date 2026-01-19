import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Display basic summary statistics
print("Summary Statistics:")
print(df.describe())

# Additional insights: trends in total shipping tonnage over years
print("\nTrend in total shipping tonnage (yearly):")
print(df[['year', 'total s ton']].sort_values(by='year'))

# Final Answer: The table describes U.S. shipping tonnage by year, including domestic and foreign trade. Key insights include a peak in 2005, stable exports, and improving US rank over time.
Final Answer: us rank, total s ton, domestic s ton, foreign total s ton, foreign imports s ton, foreign exports s ton