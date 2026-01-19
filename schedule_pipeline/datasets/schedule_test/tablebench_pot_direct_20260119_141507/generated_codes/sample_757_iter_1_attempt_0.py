import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows for seasons 1 to 6 (season column is in the 'season' column)
filtered_df = df[df['season'].between(1, 6)]
# Extract viewership values
viewership_values = filtered_df['us viewers (millions)'].astype(float)
# Calculate the average viewership
forecasted_viewership = viewership_values.mean()
print(f"Final Answer: {forecasted_viewership:.2f}")