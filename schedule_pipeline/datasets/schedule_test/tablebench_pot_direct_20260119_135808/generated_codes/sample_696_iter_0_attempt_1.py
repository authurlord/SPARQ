import pandas as pd

df = pd.read_csv('table.csv')
# Extract the 'foreign' column
foreign_data = df['foreign']
# Find the municipality with the highest and lowest values
max_municipality = df.loc[foreign_data.idxmax(), 'teoponte municipality']
min_municipality = df.loc[foreign_data.idxmin(), 'teoponte municipality']
# Calculate the difference
difference = foreign_data.max() - foreign_data.min()
print(f"Final Answer: {max_municipality}, {difference}")