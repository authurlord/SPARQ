import pandas as pd

df = pd.read_csv('table.csv')
# Filter out rows where language is 'english'
filtered_df = df[df['language'] != 'english']
# Convert 'points' to numeric, coercing errors to NaN
filtered_df['points'] = pd.to_numeric(filtered_df['points'], errors='coerce')
# Calculate the average (mean) of points for non-English entries
average_points = filtered_df['points'].mean()
print(f"Final Answer: {average_points:.1f}")