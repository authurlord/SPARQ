import pandas as pd
import numpy as np

# Load the dataframe
df = pd.read_csv('table.csv')

# Extract the relevant columns: 'Applications' and 'Offer Rate (%)'
# The years are columns: 2013 to 2017
applications = df.loc[0, '2017':'2013'].astype(float)
offer_rate = df.loc[1, '2017':'2013'].astype(float)

# Reverse the order to match the chronological order (2013 to 2017)
applications = applications[::-1]
offer_rate = offer_rate[::-1]

# Calculate the correlation coefficient
correlation_coefficient = np.corrcoef(applications, offer_rate)[0, 1]

print(f"Final Answer: {correlation_coefficient:.3f}")