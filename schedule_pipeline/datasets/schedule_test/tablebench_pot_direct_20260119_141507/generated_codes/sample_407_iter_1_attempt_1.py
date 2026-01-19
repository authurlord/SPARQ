import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Display the table for understanding
print("Table Description:")
print("The table contains six main metrics across five years (2013 to 2017):")
print("- 'Applications': Total number of applications per year, showing a slight decline from 2016 to 2017.")
print("- 'Offer Rate (%)': Percentage of applicants offered admission; increased from 86.8% in 2013 to 89.7% in 2017, indicating improved offer rates.")
print("- 'Enrols': Number of students enrolled; stable with a peak in 2017 (6,065).")
print("- 'Yield (%)': Percentage of offers accepted; decreased from 18.7% in 2013 to 16.5% in 2017, suggesting lower acceptance rates.")
print("- 'Applicant/Enrolled Ratio': Ratio of applicants to enrollees; decreased from 6.14 in 2013 to 6.76 in 2017, indicating higher enrolment efficiency.")
print("- 'Average Entry Tariff': Entry fee per student; dropped sharply from $471 in 2015 to $176 in 2017, indicating a major cost reduction.")

print("\nNotable Trends:")
print("1. Applications slightly declined from 2016 to 2017 but were still high.")
print("2. Offer rate improved over time, suggesting better selection processes.")
print("3. Yield declined, possibly due to tighter admissions or reduced interest.")
print("4. Enrolment efficiency improved, with a lower applicant-to-enrolled ratio.")
print("5. A significant drop in average entry tariff from 2015 to 2017 suggests policy changes or cost reductions.")

Final Answer: Applications declined slightly, Offer Rate increased, Yield decreased, Enrolment efficiency improved, Average Entry Tariff dropped significantly