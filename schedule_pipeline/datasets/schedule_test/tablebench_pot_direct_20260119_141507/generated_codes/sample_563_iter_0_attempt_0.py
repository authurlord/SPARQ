import pandas as pd

df = pd.read_csv('table.csv')
# Find the year with the highest number of major hurricanes
max_major_hurricanes_year = df.loc[df['number of major hurricanes'].idxmax(), 'year']
# Get the number of tropical storms for that year
tropical_storms_in_max_year = df.loc[df['number of major hurricanes'].idxmax(), 'number of tropical storms']
print(f"Final Answer: {tropical_storms_in_max_year}")