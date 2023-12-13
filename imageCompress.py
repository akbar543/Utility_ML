import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

def recreate_image(c, labels, w, h, d):
  image = np.zeros((w,h,d))

  label_idx = 0

  for i in range(w):
    for j in range(h):
      image[i][j] = c[labels[label_idx]]

      label_idx+=1

  return(image)

def compress(imagepath):
  original_img = plt.imread(imagepath)
  w,h,d = original_img.shape
  print(imagepath)
  root, extension = os.path.splitext(imagepath)
  if(extension=='.jpg'):
    original_img = original_img / 255
  X_img = np.reshape(original_img, (original_img.shape[0] * original_img.shape[1], original_img.shape[2]))
  image_array_sample = shuffle(X_img, random_state=1)[:1000]
  kmeans = KMeans(n_clusters=16, random_state=1)
  kmeans.fit(image_array_sample)
  labels = kmeans.predict(X_img)
  c = kmeans.cluster_centers_
  compressed = recreate_image(c, labels, w, h, d)
  plt.axis('off')
  plt.imshow(compressed)
  plt.savefig('static/output'+extension, bbox_inches='tight', pad_inches=0)
  n = os.path.getsize('static/output'+extension)
  o = os.path.getsize(imagepath)
  # print("n",n,"o", o)
  return (o,n)
  # plt.show()
  

