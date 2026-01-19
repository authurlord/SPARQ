import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'points' to numeric, coercing errors to NaN
df['points'] = pd.to_numeric(df['points'], errors='coerce')
# Filter rows where language is not 'english'
filtered_df = df[df['language'] != 'english']
# Calculate the average points for non-English entries
average_points = filtered_df['points'].mean()
print(f"Final Answer: {average_points:.1f}")