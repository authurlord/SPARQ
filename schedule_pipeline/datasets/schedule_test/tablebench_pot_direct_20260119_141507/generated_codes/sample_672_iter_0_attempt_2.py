import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Clean the column names to ensure correct access
df.columns = ['Bank', 'Foundation', '# of Branches As of 30 September 2012', 'Total Assets (million TL) As of 30 September 2012']

# Convert the numeric columns to integers, removing commas and handling any parsing errors
df['# of Branches As of 30 September 2012'] = df['# of Branches As of 30 September 2012'].str.replace(',', '').astype(int)
df['Total Assets (million TL) As of 30 September 2012'] = df['Total Assets (million TL) As of 30 September 2012'].str.replace(',', '').astype(int)

# Calculate the correlation coefficient
correlation = df['# of Branches As of 30 September 2012'].corr(df['Total Assets (million TL) As of 30 September 2012'])

print(f"Final Answer: {correlation:.2f}")