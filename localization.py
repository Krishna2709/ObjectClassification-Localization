# Import required modules
import tensorflow.keras as K
from data import ds

# Load the base model
base_model = K.applications.MobileNetV2(input_shape=(224,224,3), include_top=False)

# convolutional layers hyperparams
conv_opts = dict(
    activation='relu',
    padding='same',
    kernel_regularizer="l2"
)

# fine tuning the base model
x = K.layers.Conv2D(256, (1,1), **conv_opts)(base_model.output)
x = K.layers.Conv2D(256, (3,3), strides=2, **conv_opts)(x)

out = K.layers.Flatten()(x)
out = K.layers.Dense(4, activation='sigmoid')(out)

# Define the model
model = K.Model(inputs=base_model.input, outputs=out)
model.compile(loss='mean_squared_error',
                optimizer='adam',
                metrics=[K.metrics.RootMeanSquaredError(), 'mae'])

# split the dataset
inp_ds = ds.map(lambda d: (d.image,d.box)) 
valid = inp_ds.take(1000)
train = inp_ds.skip(1000).shuffle(10000)

# Training the model
checkpoint = K.callbacks.ModelCheckpoint("localization.h5", monitor='val_root_mean_squared_error', save_best_only=True, verbose=1)
model.fit(train.batch(32), epochs=12, validation_data=valid.batch(1),
            callbacks=[checkpoint])