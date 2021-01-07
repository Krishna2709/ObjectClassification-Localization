# 1. Import required modules
import tensorflow.keras as K
from data import ds

# 2. Load a pretrained model
base_model = K.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False)

print("Batch, Image Size, Channels: ", base_model.output.shape)

# Building the classifier

# freezing the layer weights
for layer in base_model.layers:
    layer.trainable = False

x = K.layers.GlobalAveragePooling2D()(base_model.output)

# print(base_model.output.shape, x.shape)

is_breeds = True
if is_breeds:
    out = K.layers.Dense(37, activation="softmax")(x)
    inp_ds = ds.map(lambda d: (d.image, d.breed))
else:
    out = K.layers.Dense(2, activation="softmax")(x)
    inp_ds = ds.map(lambda d: (d.image, d.type))

# 3. Define the model
model = K.Model(inputs=base_model.input, outputs=out)

# 4. Training and Evaluating the model
model.compile(loss="categorical_crossentropy", optimizer="adam",
                metrics=["categorical_accuracy", "top_k_categorical_accuracy"])
# split the dataset
valid = inp_ds.take(1000)
train = inp_ds.skip(1000).shuffle(10**4)

print("Training the model")
model.fit(train.batch(32), epochs=4)
print("Training completed")
print(" - - - - - - -  #### - - - - - - -")
print("Evaluating the model")

model.evaluate(valid.batch(1))

