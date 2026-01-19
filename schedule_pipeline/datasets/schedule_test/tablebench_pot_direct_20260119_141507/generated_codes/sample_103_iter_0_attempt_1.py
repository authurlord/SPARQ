import pandas as pd
import matplotlib.pyplot as plt
import re

# Load the data
df = pd.read_csv('table.csv')

# Clean the data: extract percentages from strings like '43,671 (31%)'
def extract_percentage(value):
    match = re.search(r'\((\d+)%\)', value)
    if match:
        return float(match.group(1)) / 100.0
    return 0.0

# Convert the columns to numeric for percentages
years = ['1880', '1899', '1913', '19301', '1956', '1966', '1977', '1992', '2002']
ethnicities = df['Ethnicity'].tolist()

# Prepare a list of percentage data per ethnicity
percentage_data = []
for ethnicity in ethnicities:
    row = df[df['Ethnicity'] == ethnicity].iloc[0]
    values = []
    for year in years:
        value_str = row[year]
        # Extract percentage if present, otherwise use 0
        if isinstance(value_str, str):
            pct = extract_percentage(value_str)
        else:
            pct = 0.0
        values.append(pct)
    percentage_data.append(values)

# Create a DataFrame for plotting
data_df = pd.DataFrame(percentage_data, index=ethnicities, columns=years)

# Plot stacked bar chart
plt.figure(figsize=(12, 8))
data_df.plot(kind='bar', stacked=True, color=plt.cm.viridis.colors)
plt.title('Proportion of Different Ethnic Populations Over Time')
plt.xlabel('Year')
plt.ylabel('Proportion (%)')
plt.legend(title='Ethnicity', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()