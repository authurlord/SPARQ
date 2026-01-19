import pandas as pd

df = pd.read_csv('table.csv')
# Filter out rows where '% of popular vote' is 'n.a.'
df_clean = df[df['% of popular vote'] != 'n.a.']
# Convert the percentage column to numeric
df_clean['% of popular vote'] = pd.to_numeric(df_clean['% of popular vote'], errors='coerce')
# Drop any remaining NaN values
df_clean = df_clean.dropna(subset=['% of popular vote'])
# Calculate the mean of the popular vote percentage
mean_popular_vote = df_clean['% of popular vote'].mean()
print(f"Final Answer: {mean_popular_vote:.2f}")