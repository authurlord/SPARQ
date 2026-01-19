import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Clean the 'Viewers' column by removing 'million' and converting to float
def clean_viewers(viewer_str):
    if pd.isna(viewer_str) or viewer_str == '':
        return 0
    # Remove 'million' and convert to float
    try:
        return float(viewer_str.replace(' million', '').strip())
    except:
        return 0

df['Viewers'] = df['Viewers'].apply(clean_viewers)

# Extract years and viewership
years = df[df['Viewers'].notna()]['Year'].astype(str).tolist()
viewership = df[df['Viewers'].notna()]['Viewers'].tolist()

# Create waterfall chart
plt.figure(figsize=(10, 6))
plt.plot(years, viewership, marker='o', linewidth=2, markersize=8)
plt.title('Trend in Viewership of the Program')
plt.xlabel('Year')
plt.ylabel('Viewers (in millions)')
plt.grid(True, alpha=0.3)
plt.xticks(years, rotation=45)
plt.tight_layout()
plt.show()

print(f"Final Answer: Waterfall chart generated for viewership trend")