import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('table.csv', header=None)

# Assign column names from the first row
column_names = ['-', '2017', '2016', '2015', '2014', '2013']
df.columns = column_names

# Extract the "Applications" and "Offer Rate (%)" rows
applications = df.loc[0, '2017':'2013'].astype(float)
offer_rate = df.loc[1, '2017':'2013'].astype(float)

# Remove any NaN values or invalid entries (e.g., 'n/a')
applications = applications.dropna()
offer_rate = offer_rate.dropna()

# Ensure both series have the same length
if len(applications) != len(offer_rate):
    # Pad or truncate to match, but since both come from same years, they should align
    min_len = min(len(applications), len(offer_rate))
    applications = applications[:min_len]
    offer_rate = offer_rate[:min_len]

# Compute correlation
correlation_coefficient = np.corrcoef(applications, offer_rate)[0, 1]

print(f"Final Answer: {correlation_coefficient:.3f}")