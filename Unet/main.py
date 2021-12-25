from model import *
from data import *
import tensorflow as tf

path_train = "dataset\\data_semantics\\training"
path_test  = "dataset\\data_semantics\\testing"
 
lst_train_x = [path_train+"\\semantic_rgb\\"+f for f in os.listdir(path_train+"\\image_2") if os.path.isfile(os.path.join(path_train+"\\image_2", f))]
lst_train_y = [path_train+"\\semantic_rgb\\"+f for f in os.listdir(path_train+"\\semantic_rgb") if os.path.isfile(os.path.join(path_train+"\\semantic_rgb", f))]

print(len(lst_train_x),len(lst_train_y))

dataset = load_data(lst_train_x, lst_train_y)
print(dataset[0])


# Free up RAM in case the model definition cells were run multiple times
tf.keras.backend.clear_session()

img_height = 256
img_width = 256
num_channels = 3
filters = 32
n_classes = 13

model = unet(input_size = (img_height, img_width, num_channels), filters=filters, n_classes=n_classes)

# model.summary()
# tf.keras.utils.plot_model(model, show_shapes=True)

# callback = EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True)
# reduce_lr = ReduceLROnPlateau(monitor='val_accuracy',factor=1e-1, patience=5, verbose=1, min_lr = 2e-6)
# model_checkpoint = ModelCheckpoint('unet_00.hdf5', monitor='loss',verbose=1, save_best_only=True)
# batch_size = 32
# epochs = 30
# history = model.fit(train_dataset, 
#                     validation_data = validation_dataset, 
#                     epochs = epochs, 
#                     verbose=1, 
#                     callbacks = [model_checkpoint], 
#                     batch_size = batch_size, 
#                     shuffle = True)