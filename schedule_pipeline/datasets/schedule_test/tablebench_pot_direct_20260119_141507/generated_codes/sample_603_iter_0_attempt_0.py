import pandas as pd

df = pd.read_csv('table.csv')
# Filter the row for Francisco Bravo Medical Magnet High School
school_data = df[df['School'] == 'Francisco Bravo Medical Magnet High School']
# Extract the scores for the years 2007 to 2011
scores = school_data[['2007', '2008', '2009', '2010', '2011']].apply(pd.to_numeric, errors='coerce')
# Drop NaN values if any and calculate mean
mean_score = scores.mean().mean()
print(f"Final Answer: {mean_score:.1f}")