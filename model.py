
import os
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Activation, BatchNormalization, LayerNormalization
from tensorflow.keras.layers import UpSampling2D, SeparableConv2D, Input
from tensorflow.keras.layers import GlobalAveragePooling2D, ZeroPadding2D, Cropping2D
from tensorflow.keras.layers import Add, Concatenate, Lambda, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from se import squeeze_excite_block

def residual_block(x, num_filters):
    x_init = x
    x = Conv2D(num_filters//4, (1, 1), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters//4, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)

    s = Conv2D(num_filters, (1, 1), padding="same")(x)
    s = BatchNormalization()(x)

    x = Add()([x, s])
    x = Activation("relu")(x)
    x = squeeze_excite_block(x)
    return x

def NanoNet_A(input_shape):

    f = [32, 64, 128]
    inputs = Input(shape=input_shape, name="input_image")

    ## Encoder
    encoder = MobileNetV2(input_tensor=inputs, weights="imagenet", include_top=False, alpha=0.50)
    encoder_output = encoder.get_layer(name="block_6_expand_relu").output
    skip_connections_name = ["input_image", "block_1_expand_relu", "block_3_expand_relu"]

    x = residual_block(encoder_output, 192)

    ## Decoder
    for i in range(1, len(skip_connections_name)+1, 1):
        x_skip = encoder.get_layer(skip_connections_name[-i]).output
        x_skip = Conv2D(f[-i], (1, 1), padding="same")(x_skip)
        x_skip = BatchNormalization()(x_skip)
        x_skip = Activation("relu")(x_skip)

        x = UpSampling2D((2, 2), interpolation='bilinear')(x)

        try:
            x = Concatenate()([x, x_skip])
        except Exception as e:
            x = Cropping2D(cropping=((1, 0), (0, 0)))(x)
            x = Concatenate()([x, x_skip])

        x = residual_block(x, f[-i])

    ## Output
    x = Conv2D(1, (1, 1), padding="same")(x)
    x = Activation("sigmoid")(x)

    model = Model(inputs=inputs, outputs=x)
    return model

def NanoNet_B(input_shape):
    f = [32, 64, 96]
    inputs = Input(shape=input_shape, name="input_image")

    ## Encoder
    encoder = MobileNetV2(input_tensor=inputs, weights="imagenet", include_top=False, alpha=0.35)
    encoder_output = encoder.get_layer(name="block_6_expand_relu").output
    skip_connections_name = ["input_image", "block_1_expand_relu", "block_3_expand_relu"]

    x = residual_block(encoder_output, 128)

    ## Decoder
    for i in range(1, len(skip_connections_name)+1, 1):
        x_skip = encoder.get_layer(skip_connections_name[-i]).output
        x_skip = Conv2D(f[-i], (1, 1), padding="same")(x_skip)
        x_skip = BatchNormalization()(x_skip)
        x_skip = Activation("relu")(x_skip)

        x = UpSampling2D((2, 2), interpolation='bilinear')(x)

        try:
            x = Concatenate()([x, x_skip])
        except Exception as e:
            x = Cropping2D(cropping=((1, 0), (0, 0)))(x)
            x = Concatenate()([x, x_skip])

        x = residual_block(x, f[-i])

    ## Output
    x = Conv2D(1, (1, 1), padding="same")(x)
    x = Activation("sigmoid")(x)

    model = Model(inputs=inputs, outputs=x)
    return model

def NanoNet_C(input_shape):
    f = [16, 24, 32]
    inputs = Input(shape=input_shape, name="input_image")

    ## Encoder
    encoder = MobileNetV2(input_tensor=inputs, weights="imagenet", include_top=False, alpha=0.35)
    encoder_output = encoder.get_layer(name="block_6_expand_relu").output
    skip_connections_name = ["input_image", "block_1_expand_relu", "block_3_expand_relu"]

    x = residual_block(encoder_output, 48)

    ## Decoder
    for i in range(1, len(skip_connections_name)+1, 1):
        x_skip = encoder.get_layer(skip_connections_name[-i]).output
        x_skip = Conv2D(f[-i], (1, 1), padding="same")(x_skip)
        x_skip = BatchNormalization()(x_skip)
        x_skip = Activation("relu")(x_skip)

        x = UpSampling2D((2, 2), interpolation='bilinear')(x)

        try:
            x = Concatenate()([x, x_skip])
        except Exception as e:
            x = Cropping2D(cropping=((1, 0), (0, 0)))(x)
            x = Concatenate()([x, x_skip])

        x = residual_block(x, f[-i])

    ## Output
    x = Conv2D(1, (1, 1), padding="same")(x)
    x = Activation("sigmoid")(x)

    model = Model(inputs=inputs, outputs=x)
    return model

if __name__ == "__main__":
    params = {}
    params["img_height"] = 256
    params["img_width"] = 256
    params["img_channels"] = 3
    params["mask_channels"] = 1

    model = NanoNet_C(params)
    model.summary()

    session = tf.compat.v1.Session()
    graph = tf.compat.v1.get_default_graph()

    with graph.as_default():
        with session.as_default():
            model = NanoNet_A(params)
            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()

            # Optional: save printed results to file
            flops_log_path = 'files/tf_flops_log.txt'
            opts['output'] = 'file:outfile={}'.format(flops_log_path)

            # We use the Keras session graph in the call to the profiler.
            flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd='op', options=opts)

    tf.compat.v1.reset_default_graph()
    print(flops.total_float_ops)
