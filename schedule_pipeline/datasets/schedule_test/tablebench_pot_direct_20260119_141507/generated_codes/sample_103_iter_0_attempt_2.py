import pandas as pd
import matplotlib.pyplot as plt
import re

# Load the data
df = pd.read_csv('table.csv')

# Clean the data: extract percentages from strings like '43,671 (31%)'
def extract_percentage(value):
    match = re.search(r'\((\d+)%\)', value)
    if match:
        return float(match.group(1))
    return 0.0

# Convert the data to numeric for percentages
# We will work on rows where the ethnicity is not 'All'
ethnicity_data = df[df['Ethnicity'] != 'All'].copy()

# Extract the years (columns) and clean the percentage values
years = [col for col in ethnicity_data.columns if col not in ['Ethnicity', '1880', '1899', '1913', '19301', '1956', '1966', '1977', '1992', '2002']]
# Fix typo: '19301' should be '1930'
years = [col.replace('19301', '1930') if col == '19301' else col for col in years]

# Create a list of percentage values for each ethnicity per year
percentage_data = []
for _, row in ethnicity_data.iterrows():
    year_data = {}
    for year in years:
        val_str = row[year]
        if isinstance(val_str, str):
            pct = extract_percentage(val_str)
        else:
            pct = 0.0
        year_data[year] = pct
    percentage_data.append(year_data)

# Convert to DataFrame for plotting
percentage_df = pd.DataFrame(percentage_data)
percentage_df = percentage_df.fillna(0)

# Normalize each row (year) to sum to 100%
percentage_df = percentage_df.div(percentage_df.sum(axis=1), axis=0) * 100

# Plot the stacked bar chart
plt.figure(figsize=(12, 8))
percentage_df.plot(kind='bar', stacked=True, color=plt.cm.viridis.colors)
plt.title('Proportion of Different Ethnic Populations Over Time')
plt.xlabel('Year')
plt.ylabel('Percentage (%)')
plt.xticks(rotation=45)
plt.legend(title='Ethnicity', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

print("Final Answer: Chart generated successfully")