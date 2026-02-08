#!/usr/bin/env python3
# Stacked bar chart showing quantities of different fruits per person
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

people = ['Farrah', 'Fred', 'Felicia']
fruit_names = ['apples', 'bananas', 'oranges', 'peaches']
colors = ['red', 'yellow', '#ff8000', '#ffe5b4']

x_pos = np.arange(len(people))

plt.bar(x_pos, fruit[0], width=0.5, color=colors[0], label=fruit_names[0])

plt.bar(x_pos, fruit[1], width=0.5, color=colors[1],
        label=fruit_names[1], bottom=fruit[0])

plt.bar(x_pos, fruit[2], width=0.5, color=colors[2],
        label=fruit_names[2], bottom=fruit[0] + fruit[1])

plt.bar(x_pos, fruit[3], width=0.5, color=colors[3],
        label=fruit_names[3], bottom=fruit[0] + fruit[1] + fruit[2])

plt.title('Number of Fruit per Person')
plt.ylabel('Quantity of Fruit')

plt.yticks(np.arange(0, 81, 10))

plt.xticks(x_pos, people)

plt.legend()

plt.show()
