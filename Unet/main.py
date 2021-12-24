from model import *
from data import *







train_path = ""
test_path = ""


# Free up RAM in case the model definition cells were run multiple times
tf.keras.backend.clear_session()

img_height = 256
img_width = 256
num_channels = 3
filters = 32
n_classes = 13

model = unet((img_height, img_width, num_channels), filters=32, n_classes=23)
model.summary()

tf.keras.utils.plot_model(model, show_shapes=True)

