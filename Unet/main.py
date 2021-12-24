from model import *
from data import *
import tensorflow as tf







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

# callback = EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True)
# reduce_lr = ReduceLROnPlateau(monitor='val_accuracy',factor=1e-1, patience=5, verbose=1, min_lr = 2e-6)
model_checkpoint = ModelCheckpoint('unet_00.hdf5', monitor='loss',verbose=1, save_best_only=True)
batch_size = 32
epochs = 30
history = model.fit(train_dataset, 
                    validation_data = validation_dataset, 
                    epochs = epochs, 
                    verbose=1, 
                    callbacks = [model_checkpoint], 
                    batch_size = batch_size, 
                    shuffle = True)