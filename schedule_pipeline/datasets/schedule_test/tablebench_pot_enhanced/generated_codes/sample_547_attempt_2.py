import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for Ghent
ghent_data = df[df['location'] == 'ghent']

# Separate data for 2010 and 2011
ghent_2010 = ghent_data[ghent_data['year'] == '2010']
ghent_2011 = ghent_data[ghent_data['year'] == '2011']

# Calculate average final scores
avg_2010 = ghent_2010['score - final'].astype(float).mean()
avg_2011 = ghent_2011['score - final'].astype(float).mean()

# Compute difference
difference = avg_2011 - avg_2010
print(f"Final Answer: {difference:.3f}")