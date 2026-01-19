import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for the year 1931
row_1931 = df[df['Year'] == '1931']
# Calculate R class in service at the end of 1931
r_class_end = int(row_1931['R class in service at start of year'].iloc[0]) - int(row_1931['Quantity withdrawn'].iloc[0])
print(f"Final Answer: {r_class_end}")