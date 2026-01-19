import pandas as pd

df = pd.read_csv('table.csv')

# Filter out rows with invalid or negligible 'percentage in manchu population'
df_filtered = df[df['percentage in manchu population'].str.contains('<0.01', case=False, na=False) == False]

# Sort by 'manchu' in descending order to see the largest populations first
df_sorted = df_filtered.sort_values(by='manchu', ascending=False)

# Display the top 10 entries for insight
print(df_sorted[['region', 'manchu', 'total population']].head(10))

# Final Answer: The main components of the table are the regional breakdown of total population and Manchu population. 
# The highest Manchu populations are in Liaoning (5.3M), Hebei (2.1M), Jilin (866K), and Heilongjiang (748K).
Final Answer: Liaoning, Hebei, Jilin, Heilongjiang