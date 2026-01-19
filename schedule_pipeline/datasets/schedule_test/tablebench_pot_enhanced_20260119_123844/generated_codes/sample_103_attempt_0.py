import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Clean and process the data
# Remove the percentage signs and convert to float for each column (except 'Ethnicity')
years = ['1880', '1899', '1913', '19301', '1956', '1966', '1977', '1992', '2002']
data = df[years].copy()

# Convert values to numeric, removing parentheses and percentages
for col in data.columns:
    data[col] = data[col].str.replace(r'[^\d.]', '', regex=True).astype(float)

# Set up the stacked bar chart
plt.figure(figsize=(12, 8))
bottom = 0
colors = plt.cm.Set3.colors  # Use a distinct color palette

for i, ethnicity in enumerate(df['Ethnicity']):
    if ethnicity == 'All':
        continue  # Skip 'All' since it's the total population
    plt.bar(years, data.loc[i], bottom=bottom, label=ethnicity, color=colors[i % len(colors)])
    bottom += data.loc[i]

# Add labels and title
plt.xlabel('Year')
plt.ylabel('Population')
plt.title('Proportion of Different Ethnic Populations Over Time (Stacked Bar Chart)')
plt.legend(title='Ethnicity', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()