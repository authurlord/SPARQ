import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Convert 'points' to numeric, coercing errors to NaN (but we assume all are valid)
df['points'] = pd.to_numeric(df['points'], errors='coerce')

# Compute median points of all songs
median_points = df['points'].median()

# Filter songs with non-English language and points > median
filtered_df = df[
    (df['language'] != 'english') & 
    (df['points'] > median_points)
]

# If no such songs exist, average is undefined; return 0 or handle appropriately
if filtered_df.empty:
    print("Final Answer: 0")
else:
    average_points = filtered_df['points'].mean()
    print(f"Final Answer: {average_points:.1f}")