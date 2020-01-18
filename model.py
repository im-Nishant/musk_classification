import keras
from keras.layers import *
from keras.models import *

def dense_model(
    input_shape: tuple,
    activation_func: str='relu',
    output_activation: str='relu',
    ):

    input_l = Input(
        shape=input_shape,
        name='input_l'
    )
    num_units = input_shape[0]//2
    layer_x = input_l
    while num_units>=2:
        layer_x = Dense(
            num_units,
            activation=activation_func,
            name=f'layer_with_{num_units}_units'
        )(layer_x)
        num_units=num_units//2

    output = Dense(
        1,
        activation=output_activation,
        name='output'
    )(layer_x)

    return(
        Model(inputs=[input_l], outputs=[output])
    )


def class_model(
    input_shape: tuple,
    activation_func: str='relu',
    output_activation: str='softmax'
    ):

    input_l = Input(
    shape=input_shape,
    name='input_l'
    )
    num_units = input_shape[0]//2
    layer_x = input_l
    while num_units>=4:
        layer_x = Dense(
            num_units,
            activation=activation_func,
            name=f'layer_with_{num_units}_units'
        )(layer_x)
        num_units=num_units//2
    
    output = Dense(
        2,
        activation=output_activation,
        name='output'
    )(layer_x)

    return(
        Model(inputs=[input_l], outputs=[output])
    )
