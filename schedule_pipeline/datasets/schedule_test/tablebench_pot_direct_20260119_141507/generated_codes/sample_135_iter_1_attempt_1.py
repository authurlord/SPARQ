import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Convert 'total' to numeric (in case of string)
df['total'] = pd.to_numeric(df['total'], errors='coerce')

# Compute percentage of total sales for each region per song
# We'll create a pivot table: index = song, columns = region, values = percentage
# First, create a copy and compute percentages
region_data = df[['song', 'northern ireland', 'northern england', 'scotland', 'southern england', 'wales', 'total']].copy()
region_data['total'] = region_data['total'].fillna(0)

# Calculate percentage of total sales for each region per song
percentages = []
for idx, row in region_data.iterrows():
    total = row['total']
    if total == 0:
        continue
    region_vals = [
        row['northern ireland'] / total * 100,
        row['northern england'] / total * 100,
        row['scotland'] / total * 100,
        row['southern england'] / total * 100,
        row['wales'] / total * 100
    ]
    percentages.append(region_vals)

# Now create a DataFrame for plotting
songs = region_data['song'].tolist()
regions = ['northern ireland', 'northern england', 'scotland', 'southern england', 'wales']
percentage_df = pd.DataFrame(percentages, index=songs, columns=regions)

# Plotting
plt.figure(figsize=(12, 8))
percentage_df.plot(kind='bar', width=0.8, color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'plum'])
plt.title('Percentage of Total Sales for Each Song by Region')
plt.xlabel('Song')
plt.ylabel('Percentage of Total Sales (%)')
plt.xticks(rotation=45)
plt.legend(title='Region')
plt.tight_layout()
plt.show()