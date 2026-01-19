import pandas as pd

df = pd.read_csv('table.csv')
# Find the year with the maximum number of major hurricanes
max_major_hurricanes = df['number of major hurricanes'].max()
# Get all years with that maximum
years_with_max = df[df['number of major hurricanes'] == max_major_hurricanes]
# Get the number of tropical storms for those years and take the maximum (or just one if unique)
tropical_storms_in_max_year = years_with_max['number of tropical storms'].max()
print(f"Final Answer: {tropical_storms_in_max_year}")