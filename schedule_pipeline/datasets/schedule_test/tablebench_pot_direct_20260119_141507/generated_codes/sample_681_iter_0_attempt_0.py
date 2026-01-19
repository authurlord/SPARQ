import pandas as pd

df = pd.read_csv('table.csv')
# Extract the relevant columns
pop_2010 = df['pop (2010)']
land_area = df['land ( sqmi )']

# Calculate the correlation coefficient
correlation_coefficient = pop_2010.corr(land_area)
print(f"Final Answer: {correlation_coefficient:.3f}")