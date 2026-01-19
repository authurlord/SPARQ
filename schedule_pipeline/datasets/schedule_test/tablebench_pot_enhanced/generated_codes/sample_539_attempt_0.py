import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for years 1990 to 2000
df_1990_2000 = df[(df['year'] >= '1990') & (df['year'] <= '2000')]
# Convert 'natural growth' to float
df_1990_2000['natural growth'] = pd.to_numeric(df_1990_2000['natural growth'])
# Calculate average natural growth rate
avg_growth = df_1990_2000['natural growth'].mean()
# Calculate standard deviation
std_growth = df_1990_2000['natural growth'].std()
# Define significant deviation as more than 1 standard deviation from mean
threshold = avg_growth + std_growth
# Find years where natural growth is significantly higher than average
significant_years = df_1990_2000[df_1990_2000['natural growth'] > threshold]['year']
# Output the result
print(f"Final Answer: {', '.join(significant_years)}")