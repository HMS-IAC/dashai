import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model

def get_VGG16(input_shape, num_classes, use_pretrained=True):
    if use_pretrained:
        vgg = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    else:
        vgg = VGG16(weights=None, include_top=False, input_shape=input_shape)
    
    x = Flatten()(vgg.output)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(num_classes, activation='sigmoid')(x)

    model = Model(inputs=vgg.input, outputs=x)
    
    
    return model


