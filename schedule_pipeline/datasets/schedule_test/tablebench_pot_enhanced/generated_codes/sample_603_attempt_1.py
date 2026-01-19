import pandas as pd

df = pd.read_csv('table.csv')
# Filter the row for Francisco Bravo Medical Magnet High School
school_row = df[df['School'] == 'Francisco Bravo Medical Magnet High School']
# Select the years 2007 to 2011 and convert to numeric, handling missing values
scores = pd.to_numeric(school_row[['2007', '2008', '2009', '2010', '2011']].values[0], errors='coerce')
# Calculate the average, ignoring NaN values
average_score = scores.mean()
print(f"Final Answer: {average_score:.1f}")