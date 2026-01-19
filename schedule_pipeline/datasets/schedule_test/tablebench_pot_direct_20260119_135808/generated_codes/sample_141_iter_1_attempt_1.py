import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Extract numerical values from 'specimen weight / size' column
# Split by space and take the first part (the number), then convert to float
df['specimen_weight_numeric'] = df['specimen weight / size'].str.split().str[0].astype(float)

# Use 'estimated exposure ( mrem ) / hr' for y-axis
exposure = df['estimated exposure ( mrem ) / hr'].astype(float)

# Create scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(df['specimen_weight_numeric'], exposure, color='blue')
plt.title('Relationship Between Specimen Weight/Size and Estimated Exposure (mrem/hr)')
plt.xlabel('Specimen Weight/Size (g or mm)')
plt.ylabel('Estimated Exposure (mrem/hr)')
plt.grid(True)
plt.show()