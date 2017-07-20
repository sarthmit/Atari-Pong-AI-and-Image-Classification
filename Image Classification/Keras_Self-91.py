import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, normalization, AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.constraints import maxnorm
batch_size = 128
num_classes = 10
epochs = 5000

(x_train,y_train),(x_test,y_test) = cifar10.load_data()

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()

model.add(Conv2D(32,(3,3), input_shape=x_train.shape[1:],kernel_initializer='glorot_normal', bias_initializer=keras.initializers.Constant(0.1),padding='same'))
model.add(LeakyReLU())
model.add(normalization.BatchNormalization())
model.add(Dropout(0.2))
model.add(Conv2D(32,(3,3),padding='same',kernel_initializer='glorot_normal', bias_initializer=keras.initializers.Constant(0.1)))
model.add(LeakyReLU())
model.add(normalization.BatchNormalization())
model.add(MaxPooling2D(pool_size=(3,3),strides=2))
model.add(Dropout(0.2))

model.add(Conv2D(64,(3,3),padding='same',kernel_initializer='glorot_normal', bias_initializer=keras.initializers.Constant(0.1)))
model.add(LeakyReLU())
model.add(normalization.BatchNormalization())
model.add(Conv2D(64,(3,3),padding='same',kernel_initializer='glorot_normal', bias_initializer=keras.initializers.Constant(0.1)))
model.add(LeakyReLU())
model.add(normalization.BatchNormalization())
model.add(Conv2D(64,(3,3),padding='same',kernel_initializer='glorot_normal', bias_initializer=keras.initializers.Constant(0.1)))
model.add(LeakyReLU())
model.add(normalization.BatchNormalization())
model.add(MaxPooling2D(pool_size=(3,3),strides=2))
model.add(Dropout(0.25))

#NEW EDIT
model.add(Conv2D(128,(5,5),padding='same',kernel_initializer='glorot_normal', bias_initializer=keras.initializers.Constant(0.1)))
model.add(LeakyReLU())
model.add(normalization.BatchNormalization())
model.add(Conv2D(128,(5,5),padding='same',kernel_initializer='glorot_normal', bias_initializer=keras.initializers.Constant(0.1)))
model.add(LeakyReLU())
model.add(normalization.BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides=2))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(512,W_constraint=maxnorm(3)))
model.add(LeakyReLU())
model.add(Dropout(0.25))

model.add(Dense(num_classes))
model.add(Activation('softmax'))

lr = 0.01
decay = lr/epochs

opt = keras.optimizers.SGD(lr=lr,decay=decay, momentum=0.9,nesterov=True)
#opt = keras.optimizers.Adam(0.0001)
#opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

datagen = ImageDataGenerator(rotation_range=5,width_shift_range=0.1,height_shift_range=0.1,horizontal_flip=True)

datagen.fit(x_train,True)

model.fit_generator(datagen.flow(x_train,y_train,batch_size=batch_size),steps_per_epoch=x_train.shape[0]//batch_size,epochs=epochs,validation_data=(x_test,y_test))