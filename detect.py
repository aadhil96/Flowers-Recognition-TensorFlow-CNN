from utils import load_data
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

(feature , labels) = load_data()
X_train , X_test , y_train , y_test = train_test_split( feature, labels , test_size = 0.1)

categories = ['daisy','dandelion','rose','sunflower','tulip']

model = tf.keras.models.load_model('model.h5')

#model.evaluate(X_test, y_test , verbose = 1)

prediction = model.predict(X_test)

plt.figure(figsize = (9,9))

for i in range(9):
    plt.subplot(3,3 , i+1)
    plt.imshow(X_test[i])
    plt.xlabel('Acutal :' + categories[y_test[i]] + '\n' + 'Predicted :' + 
                categories[np.argmax(prediction[i])])
    plt.xticks([])

plt.show()    