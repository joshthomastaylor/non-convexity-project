#selectionproportion.py

import numpy as np 
import matplotlib.pyplot as plt


x_array = np.arange(0.1, 1.1, 0.1)
y_array = 1/(np.sqrt(x_array))
plt.plot(x_array, y_array)
plt.show()