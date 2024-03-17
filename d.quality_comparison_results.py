import pandas as pd
import matplotlib.pyplot as plt

evaluation = pd.read_csv('report.csv')  # Use Pandas to read the CSV file

# Group the data by 'Frame' and sum the 'Pixel Count' for each group
evaluation_results = evaluation.groupby(['Frame'])['Pixel Count'].sum().reset_index()

# Create a line graph
plt.plot(evaluation_results['Frame'], evaluation_results['Pixel Count'], marker='o', linestyle='-')

# Add labels and a title for clarity
plt.xlabel('Frame')
plt.ylabel('Pixel Count')
plt.title('Line Graph of Evaluation Results')

# Add annotations to display the pixel count values on the graph
for i, count in enumerate(evaluation_results['Pixel Count']):
    plt.annotate(str(count), (evaluation_results['Frame'][i], count), textcoords="offset points", xytext=(0, 10), ha='center')

# Display the line graph
plt.show()
