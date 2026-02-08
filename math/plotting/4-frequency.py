#!/usr/bin/env python3
# Histogram of student grades with bins of size 10

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

plt.hist(student_grades, bins=range(0, 101, 10), edgecolor='black')

plt.title("Project A")
plt.xlabel("Grades")
plt.ylabel("Number of Students")

plt.xlim(0, 100)

plt.show()
