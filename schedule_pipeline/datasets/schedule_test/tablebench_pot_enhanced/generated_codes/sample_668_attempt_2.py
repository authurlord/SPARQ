import pandas as pd

df = pd.read_csv('table.csv')
# Convert the columns to numeric for calculation
territorial_waters = pd.to_numeric(df['area (km square) territorial waters'])
percentage_total_area = pd.to_numeric(df['percentage of total area (foreez)'])

# Calculate the correlation coefficient
correlation_coefficient = territorial_waters.corr(percentage_total_area)
print(f"Final Answer: {correlation_coefficient:.4f}")