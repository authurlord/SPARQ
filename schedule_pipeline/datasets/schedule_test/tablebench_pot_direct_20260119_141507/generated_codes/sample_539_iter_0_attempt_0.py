import pandas as pd

# Load the dataframe
df = pd.read_csv('table.csv')

# Filter rows for years 1990 to 2000 (excluding the summary row)
filtered_df = df[df['year'].str.contains(r'^199[0-9]|200[0-9]')]

# Convert 'natural growth' column to float (it's stored as string)
filtered_df['natural growth'] = filtered_df['natural growth'].astype(float)

# Calculate average natural growth rate between 1990 and 2000
avg_growth = filtered_df['natural growth'].mean()

# Identify years where natural growth differs from average by more than 0.5
deviations = abs(filtered_df['natural growth'] - avg_growth) > 0.5
years_outlier = filtered_df[deviations]['year'].tolist()

print(f"Final Answer: {', '.join(years_outlier)}")