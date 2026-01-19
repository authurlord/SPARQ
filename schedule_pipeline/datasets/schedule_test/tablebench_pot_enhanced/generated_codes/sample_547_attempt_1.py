import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for Ghent
ghent_data = df[df['location'] == 'ghent']
# Group by year and calculate average final score
avg_scores = ghent_data.groupby('year')['score - final'].mean()
# Calculate the difference between 2011 and 2010
difference = avg_scores['2011'] - avg_scores['2010']
print(f"Final Answer: {difference:.3f}")