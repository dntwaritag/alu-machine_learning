#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y0 = np.arange(0, 11) ** 3

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)

x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

# Create the figure and subplots
plt.figure(figsize=(12, 8))

# Plot 1
plt.subplot(3, 2, 1)
x = np.arange(0, 11)
'''Plotting the points'''
plt.plot(x, y0, 'r-')  # Plot y as a red solid line
plt.xlim(0, 10)  # Set x-axis range from 0 to 10
plt.legend()

# Plot 2
plt.subplot(3, 2, 2)
'''Plotting the scatter plots'''

plt.scatter(x1,y1,color='magenta')
plt.xlabel('Height (in)',fontsize='x-small')
plt.ylabel('Weight (lbs)',fontsize='x-small')
plt.title('Men\'s Height vs Weight',fontsize='x-small')
plt.legend()

# Plot 3
plt.subplot(3,2,3)

plt.plot(x2, y2)
plt.xlabel('Time (years)',fontsize='x-small')
plt.ylabel('Fraction Remaining',fontsize='x-small')
plt.title('Exponential Decay of C-14',fontsize='x-small')
plt.yscale('log')
plt.xlim(0, 28650)
plt.legend()

# Plot 4
plt.subplot(3, 2, 4)
plt.plot(x3, y31, linestyle='dashed', color='red', label='C-14')
plt.plot(x3, y32, color='green', label='Ra-226')
plt.xlabel('Time (years)', fontsize='x-small')
plt.ylabel('Fraction Remaining', fontsize='x-small')
plt.title('Exponential Decay of Radioactive Elements', fontsize='x-small')
plt.legend()

# Plot 5
plt.subplot(3, 1, 3)
plt.hist(student_grades, bins=[i for i in range(0, 101, 10)], edgecolor='black')
plt.xticks([i for i in range(0, 101, 10)])
plt.yticks([i for i in range(0, 31, 5)])
plt.xlabel('Grades')
plt.xlabel('Grades', fontsize='x-small')
plt.ylabel('Number of Students', fontsize='x-small')
plt.title('Project A', fontsize='x-small')

# Adjust layout
plt.tight_layout()

import matplotlib.pyplot as plt
# Title of the figure
plt.suptitle('All in One', fontsize='x-small')

# Show the plot
plt.show()
