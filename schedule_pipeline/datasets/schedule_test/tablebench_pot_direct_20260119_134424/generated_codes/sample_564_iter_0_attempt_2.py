import pandas as pd

df = pd.read_csv('table.csv')

# Filter data for years between 2004 and 2013
df_filtered = df[(df['Year'] >= '2004') & (df['Year'] <= '2013')]

# Clean 'No. of performances' column: replace '—' with NaN, '~200' with 200, etc.
df_filtered['No. of performances'] = df_filtered['No. of performances'].replace('—', None)
df_filtered['No. of performances'] = df_filtered['No. of performances'].replace('~200', 200)
df_filtered['No. of performances'] = pd.to_numeric(df_filtered['No. of performances'], errors='coerce')

# Calculate average number of performances
avg_performances = df_filtered['No. of performances'].mean()

print(f"Final Answer: {avg_performances:.1f}")