import pandas as pd

df = pd.read_csv('table.csv')
# Convert the 2009 and 2011 columns to integers
df['2009'] = df['2009'].astype(int)
df['2011'] = df['2011'].astype(int)

# Calculate improvement from 2009 to 2011
df['improvement'] = df['2011'] - df['2009']

# Exclude the 'world' row (where rank is None)
countries = df[df['rank'].notna()]

# Sort by improvement in descending order
top_countries = countries.sort_values(by='improvement', ascending=False).head(3)

# Extract country names
top_country_names = top_countries['country'].tolist()
print(f"Final Answer: {', '.join(top_country_names)}")