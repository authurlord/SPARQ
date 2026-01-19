import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Clean the 'Viewers' column: remove 'million' and convert to float
def clean_viewers(viewer_str):
    if pd.isna(viewer_str) or viewer_str == '':
        return None
    try:
        # Remove 'million' and convert to float
        cleaned = viewer_str.replace(' million', '').strip()
        return float(cleaned)
    except:
        return None

# Apply cleaning
df['Viewers'] = df['Viewers'].apply(clean_viewers)

# Drop rows where viewership is missing (NaN)
df = df.dropna(subset=['Viewers'])

# Extract years and viewership
years = df['Year'].astype(str)
viewership = df['Viewers']

# Create the waterfall chart
plt.figure(figsize=(10, 6))
plt.bar(years, viewership, color='skyblue', edgecolor='black', alpha=0.8)
plt.title('Trend in Viewership of the Program')
plt.xlabel('Year')
plt.ylabel('Viewers (in millions)')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels on top of bars
for i, v in enumerate(viewership):
    plt.text(i, v + 0.05, f'{v:.1f}M', ha='center', va='bottom', fontsize=10)

# Adjust layout and show
plt.tight_layout()
plt.show()

# Final Answer: The chart has been plotted successfully, showing the viewership trend.
Final Answer: chart_plotted