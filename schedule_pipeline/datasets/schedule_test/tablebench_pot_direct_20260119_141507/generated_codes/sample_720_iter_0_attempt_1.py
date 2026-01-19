import pandas as pd

df = pd.read_csv('table.csv')
# Filter out the 'world' row and drop rows where country is None
df = df[df['country'].notna()]
df = df[df['country'] != 'world']

# Convert the numeric columns to integers
df[['2009', '2010', '2011']] = df[['2009', '2010', '2011']].astype(int)

# Calculate improvement from 2009 to 2011
df['improvement'] = df['2011'] - df['2009']

# Sort by improvement in descending order and get top 3 countries
top_3_improvement = df.sort_values(by='improvement', ascending=False).head(3)['country'].tolist()

print(f"Final Answer: {', '.join(top_3_improvement)}")