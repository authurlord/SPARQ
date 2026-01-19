import pandas as pd

df = pd.read_csv('table.csv')
# Convert columns to numeric
df[['2009', '2011']] = df[['2009', '2011']].astype(int)
# Calculate improvement from 2009 to 2011
df['improvement'] = df['2011'] - df['2009']
# Exclude the 'world' row
df_countries = df[df['country'] != 'world']
# Sort by improvement in descending order and get top 3
top_3 = df_countries.nlargest(3, 'improvement')['country'].tolist()
print(f"Final Answer: {top_3[0]}, {top_3[1]}, {top_3[2]}")