import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

CATEGORIES = ["Dog", "Cat"]

def prepare(filepath):
    IMG_SIZE = 50
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    plt.imshow(img_array, cmap="gray")
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

model = tf.keras.models.load_model("Cats_vs_dogs-3-conv-64-nodes-0-dense-1551659397")


prediction = model.predict([prepare('cat5.jpg')])

print(CATEGORIES[int(prediction[0][0])])
plt.show()
