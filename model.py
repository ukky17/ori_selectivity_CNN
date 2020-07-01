from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, Dense, Activation, Flatten

def create_model(input_shape, num_classes=10):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), strides=(1, 1), activation='relu', name='layer1')(inputs)
    x = BatchNormalization(name='bn1')(x)
    x = Conv2D(32, (3, 3), strides=(2, 2), activation='relu', name='layer2')(x)
    x = BatchNormalization(name='bn2')(x)
    x = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', name='layer3')(x)
    x = BatchNormalization(name='bn3')(x)
    x = Conv2D(64, (3, 3), strides=(2, 2), activation='relu', name='layer4')(x)
    x = BatchNormalization(name='bn4')(x)
    x = Conv2D(128, (3, 3), strides=(1, 1), activation='relu', name='layer5')(x)
    x = BatchNormalization(name='bn5')(x)
    x = Conv2D(128, (3, 3), strides=(2, 2), activation='relu', name='layer6')(x)
    x = BatchNormalization(name='bn6')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(512, activation='relu', name='fc1')(x)
    x = BatchNormalization(name='bn7')(x)
    x = Dense(num_classes, name='fc2')(x)
    predictions = Activation('softmax')(x)
    model = Model(outputs=predictions, inputs=inputs)
    return model
