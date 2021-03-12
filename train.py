from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers
from keras import backend as K
from extract import X_train, X_test, Y_train, Y_test
# from extract import pipe


def Convolution(input_tensor, filters):
    x = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', strides=(1, 1))(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def model(input_shape):
    inputs = Input((input_shape))

    conv_1 = Convolution(inputs, 32)
    maxp_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv_1)

    conv_2 = Convolution(maxp_1, 64)
    maxp_2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv_2)

    conv_3 = Convolution(maxp_2, 128)
    maxp_3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv_3)

    conv_4 = Convolution(maxp_3, 256)
    maxp_4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv_4)

    conv_5 = Convolution(maxp_4, 512)
    upsample_6 = UpSampling2D((2, 2))(conv_5)

    conv_6 = Convolution(upsample_6, 256)
    upsample_7 = UpSampling2D((2, 2))(conv_6)

    upsample_7 = concatenate([upsample_7, conv_3])

    conv_7 = Convolution(upsample_7, 128)
    upsample_8 = UpSampling2D((2, 2))(conv_7)

    conv_8 = Convolution(upsample_8, 64)
    upsample_9 = UpSampling2D((2, 2))(conv_8)

    upsample_9 = concatenate([upsample_9, conv_1])

    conv_9 = Convolution(upsample_9, 32)
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(conv_9)

    mod = Model(inputs=[inputs], outputs=[outputs])
    return mod


model = model(input_shape=(240, 240, 1))
model.summary()


# Computing dice coef
def dice_coef(y_true, y_pred, smooth=1.0):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


# Computing Precision
def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


# Computing Sensitivity
def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


# Computing Specificity
def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())


# Compiling the model
Adam = optimizers.Adam(lr=0.001)
model.compile(optimizer=Adam, loss='binary_crossentropy',
              metrics=['accuracy', dice_coef, precision, sensitivity, specificity])

checkpoint = ModelCheckpoint('models/model-{epoch:02d}.h5', monitor='val_loss', verbose=0, save_best_only=True,
                             mode='auto')


# X_train, X_test, Y_train, Y_test = pipe.Data()
# Fitting the model over the data
history = model.fit(X_train, Y_train, batch_size=32, epochs=10, validation_split=0.20, verbose=1, initial_epoch=0,
                    callbacks=[checkpoint])
