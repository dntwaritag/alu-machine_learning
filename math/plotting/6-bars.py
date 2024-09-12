#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

'''This script plots a stackgraph'''
np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))  # Transpose the fruit array

labels = ['Farrah', 'Fred', 'Felicia']
fruits = ['Apples', 'Bananas', 'Oranges', 'Peaches']
colors = ['red', 'yellow', 'orange', '#ffe5b4']

fig, ax = plt.subplots()
width = 0.5

for i, fruit_name in enumerate(fruits):
    bottom = np.sum(fruit[:i], axis=0)
    ax.bar(labels, fruit[i], width, label=fruit_name, color=colors[i], bottom=bottom)

ax.set_ylabel('Quantity of Fruit')
ax.set_title('Number of Fruit per Person')
ax.set_yticks(np.arange(0, 81, 10))
ax.legend()

plt.show()
