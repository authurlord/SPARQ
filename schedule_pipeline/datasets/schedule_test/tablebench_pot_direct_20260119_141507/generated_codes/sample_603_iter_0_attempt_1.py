import pandas as pd

df = pd.read_csv('table.csv')
# Filter the row for Francisco Bravo Medical Magnet High School
school_data = df[df['School'] == 'Francisco Bravo Medical Magnet High School']
# Extract the scores from 2007 to 2011
scores = school_data[['2007', '2008', '2009', '2010', '2011']].dropna().values.flatten()
# Calculate the mean
average_score = scores.mean()
print(f"Final Answer: {average_score:.1f}")