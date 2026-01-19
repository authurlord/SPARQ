import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where 'crime' is a violent crime type
violent_crime_types = df[df['crime'].str.contains('murder|rape|robbery|aggravated assault|violent crime', case=False, na=False)]
# Count the number of such rows
num_violent_crime_types = len(violent_crime_types)
print(f"Final Answer: {num_violent_crime_types}")