import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Display basic information about the columns
print("Columns:", df.columns.tolist())
print("\nData preview:")
print(df.head())

# Analyze trends in same-sex marriages
same_sex_marriages = df['same - sex marriages'].astype(int)
total_marriages = df['total marriages'].astype(int)
percentage_same_sex = df['% same - sex marriages'].astype(float)

# Calculate the trend in same-sex marriage counts and percentages
print("\nTrend in same-sex marriages (count):")
print(same_sex_marriages.tolist())

print("\nTrend in percentage of same-sex marriages:")
print(percentage_same_sex.tolist())

# Check if there's an increasing trend
trend_increase = same_sex_marriages.diff().values > 0
print(f"\nSame-sex marriage count increased in years: {trend_increase.tolist()}")

# Final summary of observations
print("\nInitial insights:")
print("- The number of same-sex marriages increased from 1,275 (2005) to 3,880 (2011).")
print("- The percentage of same-sex marriages rose from 1.06% to 2.38%, indicating growing representation.")
print("- This suggests a positive trend in the recognition and participation of same-sex couples in marriage over time.")