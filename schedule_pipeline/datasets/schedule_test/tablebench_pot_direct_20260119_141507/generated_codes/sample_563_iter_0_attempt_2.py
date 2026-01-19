import pandas as pd

df = pd.read_csv('table.csv')

# Find the year with the highest number of major hurricanes
max_major_hurricanes_year = df.loc[df['number of major hurricanes'].idxmax(), 'year']

# Get the number of tropical storms in that year
tropical_storms_in_max_year = df.loc[df['number of major hurricanes'] == df['number of major hurricanes'].max(), 'number of tropical storms'].values[0]

print(f"Final Answer: {tropical_storms_in_max_year}")