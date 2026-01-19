import pandas as pd

df = pd.read_csv('table.csv')

# Display basic information about the table
print("Table Columns:", df.columns.tolist())
print("\nSample Data:")
print(df.head())

# Extract same-sex marriage data and calculate trend
same_sex_marriages = df['same - sex marriages'].astype(int)
percentage_same_sex = df['% same - sex marriages'].astype(float)

# Show the trend in same-sex marriages and percentages
print("\nTrend in same-sex marriages over the years:")
for idx, row in df.iterrows():
    print(f"{row['year']}: {row['same - sex marriages']} same-sex marriages ({row['% same - sex marriages']}%)")

# Final insight summary
final_insight = (
    "The table tracks annual marriages, showing that the number of same-sex marriages "
    "increased from 1,275 in 2005 to 3,880 in 2011, and the percentage of same-sex marriages "
    "rose from 1.06% to 2.38%, indicating a growing share in total marriages over time."
)

print(final_insight)