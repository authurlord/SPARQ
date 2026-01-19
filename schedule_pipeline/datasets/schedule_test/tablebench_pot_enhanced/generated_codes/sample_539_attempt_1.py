import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'year' to numeric for filtering
df['year'] = pd.to_numeric(df['year'], errors='coerce')

# Filter data for 1990 to 2000
df_1990_2000 = df[(df['year'] >= 1990) & (df['year'] <= 2000)]

# Calculate average and standard deviation of natural growth
mean_growth = df_1990_2000['natural growth'].astype(float).mean()
std_growth = df_1990_2000['natural growth'].astype(float).std()

# Define threshold for "significant" deviation (1 standard deviation)
threshold = std_growth

# Find years where natural growth is significantly different
outliers = df_1990_2000[
    (df_1990_2000['natural growth'].astype(float) > mean_growth + threshold) |
    (df_1990_2000['natural growth'].astype(float) < mean_growth - threshold)
]

# Extract the years
outlier_years = outliers['year'].tolist()

# Print final answer
print(f"Final Answer: {', '.join(map(str, outlier_years))}")