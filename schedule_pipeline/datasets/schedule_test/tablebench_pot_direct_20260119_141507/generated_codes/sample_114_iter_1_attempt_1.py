import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
df = pd.read_csv('table.csv')

# Filter the 2006–07 season data for Philadelphia
# Find the row where 'Team' is 'Philadelphia' and 'Year' is '2006–07'
philadelphia_2007 = df[(df['Year'] == '2006–07') & (df['Team'] == 'Philadelphia')]

# Extract the required values
if not philadelphia_2007.empty:
    rpg = float(philadelphia_2007['RPG'].values[0])
    apg = float(philadelphia_2007['APG'].values[0])
    spg = float(philadelphia_2007['SPG'].values[0])
    bpg = float(philadelphia_2007['BPG'].values[0])

    # Define the labels
    labels = ['Rebounds', 'Assists', 'Steals', 'Blocks']
    values = [rpg, apg, spg, bpg]

    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()

    # Close the plot
    values += values[:1]
    angles += angles[:1]

    # Create the radar chart
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='b', alpha=0.25)
    ax.plot(angles, values, color='b', linewidth=2, linestyle='solid')

    # Set the labels
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    # Title
    plt.title('Average Rebounds, Assists, Steals, and Blocks per Game (2006–07, Philadelphia)', pad=20)

    # Show the plot
    plt.show()
else:
    print("No data found for Philadelphia in 2006–07 season.")