import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Clean the columns: remove commas and convert to numeric
df['# of Branches As of 30 September 2012'] = df['# of Branches As of 30 September 2012'].str.replace(',', '').astype(float)
df['Total Assets (million TL) As of 30 September 2012'] = df['Total Assets (million TL) As of 30 September 2012'].str.replace(',', '').astype(float)

# Calculate the correlation coefficient
correlation_coefficient = df['# of Branches As of 30 September 2012'].corr(df['Total Assets (million TL) As of 30 September 2012'])

print(f"Final Answer: {correlation_coefficient:.2f}")