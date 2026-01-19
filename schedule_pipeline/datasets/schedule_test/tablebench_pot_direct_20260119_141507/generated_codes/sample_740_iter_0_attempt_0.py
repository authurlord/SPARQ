import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('table.csv')

# Convert the 'Period' column to numeric for easier analysis
df['Period'] = df['Period'].str.extract(r'(\d{4}-\d{4})')[0].astype(int) // 1000  # Extract decade (e.g., 1950->5, 2010->10)

# Select relevant columns
columns = ['CBR1', 'CDR1', 'IMR1']
data = df[columns].copy()

# Project for 2010–2015 (which corresponds to decade 2010)
# Since data ends at 2005–2010 (decade 10), we assume linear trend from 2005–2010 to 2010–2015

# Extract the last 3 years (2005–2010, 2000–2005, 1995–2000) to fit a trend
recent_data = data.iloc[-3:].copy()
recent_years = [10, 9, 8]  # corresponding to 2005–2010, 2000–2005, 1995–2000

# Fit linear trend for each rate
def project_trend(x, y, future_year):
    # x: years (as integers), y: values
    slope = np.polyfit(x, y, 1)[0]
    intercept = np.polyfit(x, y, 1)[1]
    return slope * future_year + intercept

# Project for 2010–2015 (year 11)
projected_cbr = project_trend(recent_years, data['CBR1'].iloc[-3:], 11)
projected_cdr = project_trend(recent_years, data['CDR1'].iloc[-3:], 11)
projected_imr = project_trend(recent_years, data['IMR1'].iloc[-3:], 11)

print(f"Final Answer: {projected_cbr:.2f}, {projected_cdr:.2f}, {projected_imr:.2f}")