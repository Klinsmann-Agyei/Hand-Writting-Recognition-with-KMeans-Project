import numpy as np
from matplotlib import pyplot as plt
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans

digits = datasets.load_digits()
#print(digits.DESCR)
#print(digits.data)
print(digits.target)
plt.gray() 
 
plt.matshow(digits.images[100])
 
plt.show()

print(digits.target[100])


model = KMeans(n_clusters=10, random_state=42)
model.fit(digits.data)
fig = plt.figure(figsize=(8, 3))
 
fig.suptitle('Cluser Center Images', fontsize=14, fontweight='bold')

for i in range(10):
 
  # Initialize subplots in a grid of 2X5, at i+1th position
  ax = fig.add_subplot(2, 5, 1 + i)
 
  # Display images
  ax.imshow(model.cluster_centers_[i].reshape((8, 8)), cmap=plt.cm.binary)
plt.show()

new_samples = np.array([
[0.00,0.00,4.19,7.62,7.63,6.48,0.46,0.00,0.00,4.50,7.62,4.57,1.83,7.47,4.19,0.00,0.00,7.17,4.19,0.00,0.00,5.65,6.10,0.00,0.00,0.00,0.00,0.99,3.13,7.32,5.19,0.00,0.00,0.00,1.37,7.40,7.62,6.33,1.07,0.00,0.00,0.00,3.13,7.62,7.62,7.62,7.63,1.83,0.00,0.00,2.59,5.64,3.20,3.05,3.66,0.69,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00],
[0.00,0.00,0.00,0.23,1.15,0.00,0.00,0.00,0.00,2.59,7.17,3.20,7.55,0.00,0.00,0.00,0.00,3.96,7.62,4.66,7.02,0.00,0.00,0.00,0.00,6.18,6.10,5.34,7.02,5.80,7.17,0.00,0.99,7.62,6.48,7.32,7.62,6.41,3.28,0.00,2.67,7.62,7.47,7.32,4.96,0.00,0.00,0.00,1.98,5.65,0.99,3.82,2.75,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00],
[0.00,0.00,0.00,1.60,2.29,0.00,0.00,0.00,0.00,0.00,2.67,7.63,6.41,0.00,0.00,0.00,0.00,0.91,7.02,5.95,0.38,0.00,0.00,0.00,0.00,5.42,7.17,0.99,0.00,0.00,0.00,0.00,0.00,6.86,6.48,4.20,0.99,0.00,0.00,0.00,0.00,6.41,7.62,7.62,7.17,0.00,0.00,0.00,0.00,1.45,6.49,7.62,6.33,0.00,0.00,0.00,0.00,0.00,0.23,1.53,0.53,0.00,0.00,0.00],
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.08,6.71,7.62,7.40,0.54,0.00,0.00,0.00,1.83,7.62,7.25,7.02,0.08,0.00,0.00,0.00,2.06,7.62,7.63,7.55,7.47,4.73,0.00,0.00,0.00,3.43,7.62,6.24,6.64,6.10,0.00,0.00,0.00,1.53,7.62,3.43,6.86,5.49,0.00,0.00,0.00,1.45,7.62,7.62,7.55,2.21,0.00,0.00,0.00,0.00,2.44,2.67,1.14,0.00,0.00]
])
new_labels = model.predict(new_samples)
 
print(new_labels)
for i in range(len(new_labels)):
  if new_labels[i] == 0:
    print(0, end='')
  elif new_labels[i] == 1:
    print(9, end='')
  elif new_labels[i] == 2:
    print(2, end='')
  elif new_labels[i] == 3:
    print(1, end='')
  elif new_labels[i] == 4:
    print(6, end='')
  elif new_labels[i] == 5:
    print(8, end='')
  elif new_labels[i] == 6:
    print(4, end='')
  elif new_labels[i] == 7:
    print(5, end='')
  elif new_labels[i] == 8:
    print(7, end='')
  elif new_labels[i] == 9:
    print(3, end='')
  else:
      pass

