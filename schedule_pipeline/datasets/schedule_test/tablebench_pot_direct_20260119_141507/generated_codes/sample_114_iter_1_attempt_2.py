import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
df = pd.read_csv('table.csv')

# Filter rows for the 2006–07 season
season_2006_07 = df[df['Year'] == '2006–07']

# Select only the relevant columns: RPG, APG, SPG, BPG
stats = season_2006_07[['RPG', 'APG', 'SPG', 'BPG']].dropna()

# If no valid data, raise an error
if stats.empty:
    print("No valid data found for 2006–07 season.")
else:
    # Compute average per game stats across both teams
    avg_stats = stats.mean()

    # Prepare labels and values for radar chart
    labels = ['Rebounds (RPG)', 'Assists (APG)', 'Steals (SPG)', 'Blocks (BPG)']
    values = avg_stats.values

    # Number of variables
    num_vars = len(labels)

    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # Complete the loop
    values += values[:1]
    angles += angles[:1]

    # Create the radar chart
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='blue', alpha=0.25)
    ax.plot(angles, values, color='blue', linewidth=2, linestyle='solid')
    
    # Add labels
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    # Add title
    plt.title('Average Rebounds, Assists, Steals, and Blocks per Game (2006–07 Season)', pad=20)

    # Show the plot
    plt.show()