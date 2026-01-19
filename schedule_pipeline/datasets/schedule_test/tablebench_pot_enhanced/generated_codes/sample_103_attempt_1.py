import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Clean the data: Remove '%' and convert to numeric for percentages
# We'll focus on ethnicities excluding 'All' and extract the percentage values
ethnicities = df[df['Ethnicity'] != 'All'].copy()

# Extract the years (columns) and clean the data
years = [col for col in df.columns if col not in ['Ethnicity', '1880', '1899', '1913', '19301', '1956', '1966', '1977', '1992', '2002']]
# Correcting year labels
year_labels = ['1880', '1899', '1913', '1930', '1956', '1966', '1977', '1992', '2002']

# Create a dictionary to store percentage values for each ethnicity over time
data = {}
for idx, row in ethnicities.iterrows():
    ethnicity = row['Ethnicity']
    percentages = []
    for year in year_labels:
        cell = row[year]
        # Extract percentage value (e.g., "43,671 (31%)")
        if isinstance(cell, str):
            try:
                percent = float(cell.split('(')[-1].strip('%)'))
            except:
                percent = 0.0
        else:
            percent = 0.0
        percentages.append(percent)
    data[ethnicity] = percentages

# Convert to DataFrame
percent_df = pd.DataFrame(data, index=year_labels)

# Plot stacked bar chart
plt.figure(figsize=(12, 8))
percent_df.plot(kind='bar', stacked=True, figsize=(12, 8), colormap='tab20')

plt.title('Proportion of Different Ethnic Populations Over Time')
plt.xlabel('Year')
plt.ylabel('Percentage (%)')
plt.xticks(rotation=45)
plt.legend(title='Ethnicity', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()