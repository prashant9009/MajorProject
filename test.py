from keras.models import Model, load_model
import extract

model.load_weights('/models/model-04.h5')
fig = plt.figure(figsize=(5, 5))
immmg = X_train[250, :, :]
imgplot = plt.imshow(immmg)
plt.show()