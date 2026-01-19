import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where 'Fuel Type' is 'diesel'
diesel_cars = df[df['Fuel Type'] == 'diesel']
# Count the number of diesel cars
num_diesel_cars = len(diesel_cars)
print(f"Final Answer: {num_diesel_cars}")