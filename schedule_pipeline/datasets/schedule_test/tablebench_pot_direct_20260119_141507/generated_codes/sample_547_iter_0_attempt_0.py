import pandas as pd

df = pd.read_csv('table.csv')

# Filter rows where location is 'ghent'
ghent_data = df[df['location'] == 'ghent']

# Extract final scores for 2010 and 2011
ghent_2010 = ghent_data[ghent_data['year'] == '2010']['score - final'].astype(float)
ghent_2011 = ghent_data[ghent_data['year'] == '2011']['score - final'].astype(float)

# Calculate average final scores
avg_2010 = ghent_2010.mean()
avg_2011 = ghent_2011.mean()

# Compute the difference
difference = avg_2011 - avg_2010
print(f"Final Answer: {difference:.3f}")