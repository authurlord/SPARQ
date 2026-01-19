import pandas as pd

df = pd.read_csv('table.csv')
# Count the number of diesel cars
diesel_count = df[df['Fuel Type'] == 'diesel'].shape[0]
print(f"Final Answer: {diesel_count}")