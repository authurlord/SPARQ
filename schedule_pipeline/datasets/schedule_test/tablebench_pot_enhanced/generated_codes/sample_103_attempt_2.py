import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Clean and process the data
# Remove the percentage signs and convert to float
columns = ['1880', '1899', '1913', '19301', '1956', '1966', '1977', '1992', '2002']
data_cleaned = df[columns].copy()

for col in columns:
    data_cleaned[col] = data_cleaned[col].str.replace(r'[^\d.]', '', regex=True).astype(float)

# Set up the stacked bar chart
plt.figure(figsize=(12, 8))
bottom = 0

# Define colors for each ethnicity
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8E8', '#F7DC6F', '#BB8FCE', '#85C1E9']

# Plot each ethnicity as a stacked bar
for idx, row in data_cleaned.iterrows():
    ethnicity = df.iloc[idx]['Ethnicity']
    if ethnicity == 'All':
        continue  # Skip the 'All' row as it's the total
    plt.bar(df['1880'].index, row.values, bottom=bottom, label=ethnicity, color=colors[idx % len(colors)])
    bottom += row.values

# Set labels and title
plt.xlabel('Year')
plt.ylabel('Population')
plt.title('Proportion of Ethnic Populations Over Time (Stacked Bar Chart)')
plt.xticks(df['1880'].index, df['1880'], rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Adjust layout and display
plt.tight_layout()
plt.show()