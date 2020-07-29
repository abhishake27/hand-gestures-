from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense,Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

image_size = [224,224]

vgg = VGG16(input_shape= image_size+[3],include_top=False,
                      weights="imagenet")

for layer in vgg.layers:
    layer.trainable = False


flatten = Flatten()
new_layer = Dense(3,activation='softmax')(flatten(vgg.output))


MyModel = Model(inputs=vgg.input, outputs=new_layer)
MyModel.summary()

MyModel.compile(
    loss = 'categorical_crossentropy',
    optimizer = Adam(0.0007),
    metrics = ['accuracy']
    )

    
train_data = ImageDataGenerator(rescale = 1./255,rotation_range=40,shear_range=0.2,
                                horizontal_flip=True,zoom_range=0.5).flow_from_directory('F:/rockpapersc/train_data',
                                                                            (224,224),batch_size = 50)

valid_data = ImageDataGenerator(rescale = 1./255).flow_from_directory('F:/rockpapersc/valid_data',(224,224),
                                                      batch_size = 32)

r=MyModel.fit_generator(
  train_data,
  validation_data=valid_data,
  epochs=10,
  steps_per_epoch=len(train_data),
  validation_steps=len(valid_data)
)

MyModel.save('F:/spycodes/MyModel.h5')



