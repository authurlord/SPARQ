import pandas as pd

df = pd.read_csv('table.csv')

# Filter for European Championships and 400 m event
filtered_df = df[(df['Competition'] == 'European Championships') & (df['Event'] == '400 m')]

# Convert Position to numeric by extracting the number from strings like '3rd', '17th', etc.
filtered_df['Position_Num'] = filtered_df['Position'].str.extract('(\d+)').astype(int)

# Find the row with the best (minimum) position
best_row = filtered_df.loc[filtered_df['Position_Num'].idxmin()]

# Extract the year
best_year = best_row['Year']
print(f"Final Answer: {best_year}")