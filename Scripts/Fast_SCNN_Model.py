import tensorflow as tf
from keras import layers, Model


# =========================================================
# Basic Blocks
# =========================================================
def conv_bn_relu(x, filters, kernel_size=3, strides=1, name=None):
    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        name=None if name is None else f"{name}_conv",
    )(x)
    x = layers.BatchNormalization(name=None if name is None else f"{name}_bn")(x)
    x = layers.ReLU(name=None if name is None else f"{name}_relu")(x)
    return x


def ds_conv_bn_relu(x, filters, kernel_size=3, strides=1, name=None):
    x = layers.DepthwiseConv2D(
        kernel_size,
        strides=strides,
        padding="same",
        use_bias=False,
        depthwise_initializer="he_normal",
        name=None if name is None else f"{name}_dwconv",
    )(x)
    x = layers.BatchNormalization(name=None if name is None else f"{name}_dw_bn")(x)
    x = layers.ReLU(name=None if name is None else f"{name}_dw_relu")(x)

    x = layers.Conv2D(
        filters,
        kernel_size=1,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        name=None if name is None else f"{name}_pwconv",
    )(x)
    x = layers.BatchNormalization(name=None if name is None else f"{name}_pw_bn")(x)
    x = layers.ReLU(name=None if name is None else f"{name}_pw_relu")(x)
    return x


def bottleneck_block(x, out_channels, expansion=6, strides=1, name=None):
    """
    MobileNetV2-style inverted residual bottleneck.
    """
    in_channels = x.shape[-1]
    expanded_channels = int(in_channels * expansion)

    shortcut = x

    # Expand
    y = layers.Conv2D(
        expanded_channels,
        kernel_size=1,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        name=None if name is None else f"{name}_expand_conv",
    )(x)
    y = layers.BatchNormalization(name=None if name is None else f"{name}_expand_bn")(y)
    y = layers.ReLU(name=None if name is None else f"{name}_expand_relu")(y)

    # Depthwise
    y = layers.DepthwiseConv2D(
        kernel_size=3,
        strides=strides,
        padding="same",
        use_bias=False,
        depthwise_initializer="he_normal",
        name=None if name is None else f"{name}_dwconv",
    )(y)
    y = layers.BatchNormalization(name=None if name is None else f"{name}_dw_bn")(y)
    y = layers.ReLU(name=None if name is None else f"{name}_dw_relu")(y)

    # Project
    y = layers.Conv2D(
        out_channels,
        kernel_size=1,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        name=None if name is None else f"{name}_project_conv",
    )(y)
    y = layers.BatchNormalization(name=None if name is None else f"{name}_project_bn")(y)

    # Residual
    if strides == 1 and in_channels == out_channels:
        y = layers.Add(name=None if name is None else f"{name}_add")([shortcut, y])

    return y


def pyramid_pooling_module(x, out_channels, bin_sizes=(1, 2, 3, 6), name=None):
    """
    Lightweight pyramid pooling.
    """
    h = tf.keras.backend.int_shape(x)[1]
    w = tf.keras.backend.int_shape(x)[2]

    pooled_outputs = [x]

    for i, bin_size in enumerate(bin_sizes):
        pooled = layers.AveragePooling2D(
            pool_size=(max(1, h // bin_size), max(1, w // bin_size)),
            strides=(max(1, h // bin_size), max(1, w // bin_size)),
            padding="same",
            name=None if name is None else f"{name}_pool_{i}",
        )(x)
        pooled = conv_bn_relu(
            pooled,
            out_channels,
            kernel_size=1,
            strides=1,
            name=None if name is None else f"{name}_conv_{i}",
        )
        pooled = layers.UpSampling2D(
            size=(max(1, h // pooled.shape[1]), max(1, w // pooled.shape[2])),
            interpolation="bilinear",
            name=None if name is None else f"{name}_up_{i}",
        )(pooled)
        pooled_outputs.append(pooled)

    x = layers.Concatenate(axis=-1, name=None if name is None else f"{name}_concat")(pooled_outputs)
    x = conv_bn_relu(x, out_channels, kernel_size=1, strides=1, name=None if name is None else f"{name}_out")
    return x


# =========================================================
# Learning to Downsample
# =========================================================
def learning_to_downsample(inputs, down_channels=(32, 48, 64), name="ltd"):
    c1, c2, c3 = down_channels
    x = conv_bn_relu(inputs, c1, kernel_size=3, strides=2, name=f"{name}_conv1")    # /2
    x = ds_conv_bn_relu(x, c2, kernel_size=3, strides=2, name=f"{name}_dsconv2")     # /4
    x = ds_conv_bn_relu(x, c3, kernel_size=3, strides=2, name=f"{name}_dsconv3")     # /8
    return x


# =========================================================
# Global Feature Extractor
# =========================================================
def global_feature_extractor(
    x,
    block_channels=(64, 96, 128),
    bottleneck_counts=(3, 3, 3),
    expansion=6,
    ppm_channels=128,
    name="gfe"
):
    # stage 1: /8 -> /16
    for i in range(bottleneck_counts[0]):
        stride = 2 if i == 0 else 1
        x = bottleneck_block(
            x,
            out_channels=block_channels[0],
            expansion=expansion,
            strides=stride,
            name=f"{name}_stage1_block{i+1}"
        )

    # stage 2: /16 -> /32
    for i in range(bottleneck_counts[1]):
        stride = 2 if i == 0 else 1
        x = bottleneck_block(
            x,
            out_channels=block_channels[1],
            expansion=expansion,
            strides=stride,
            name=f"{name}_stage2_block{i+1}"
        )

    # stage 3: stay /32
    for i in range(bottleneck_counts[2]):
        stride = 1
        x = bottleneck_block(
            x,
            out_channels=block_channels[2],
            expansion=expansion,
            strides=stride,
            name=f"{name}_stage3_block{i+1}"
        )

    x = pyramid_pooling_module(x, out_channels=ppm_channels, name=f"{name}_ppm")
    return x


# =========================================================
# Feature Fusion Module
# =========================================================
def feature_fusion_module(high_res, low_res, out_channels=128, name="ffm"):
    # low_res: /32 -> /8
    low_res = layers.UpSampling2D(size=(4, 4), interpolation="bilinear", name=f"{name}_low_up")(low_res)

    low_res = layers.DepthwiseConv2D(
        kernel_size=3,
        strides=1,
        padding="same",
        use_bias=False,
        depthwise_initializer="he_normal",
        name=f"{name}_low_dwconv",
    )(low_res)
    low_res = layers.BatchNormalization(name=f"{name}_low_dw_bn")(low_res)
    low_res = layers.Conv2D(
        out_channels,
        kernel_size=1,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        name=f"{name}_low_pwconv",
    )(low_res)
    low_res = layers.BatchNormalization(name=f"{name}_low_pw_bn")(low_res)

    high_res = layers.Conv2D(
        out_channels,
        kernel_size=1,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        name=f"{name}_high_conv",
    )(high_res)
    high_res = layers.BatchNormalization(name=f"{name}_high_bn")(high_res)

    x = layers.Add(name=f"{name}_add")([high_res, low_res])
    x = layers.ReLU(name=f"{name}_relu")(x)
    return x


# =========================================================
# Classifier Head
# =========================================================
def classifier_head(x, num_classes, classifier_channels=128, dropout_rate=0.1, name="classifier"):
    x = ds_conv_bn_relu(x, classifier_channels, kernel_size=3, strides=1, name=f"{name}_dsconv1")
    x = ds_conv_bn_relu(x, classifier_channels, kernel_size=3, strides=1, name=f"{name}_dsconv2")
    x = layers.Dropout(dropout_rate, name=f"{name}_dropout")(x)
    x = layers.Conv2D(
        num_classes,
        kernel_size=1,
        padding="same",
        kernel_initializer="he_normal",
        name=f"{name}_logits",
    )(x)

    # /8 -> full resolution
    x = layers.UpSampling2D(size=(8, 8), interpolation="bilinear", name=f"{name}_upsample")(x)

    if num_classes == 1:
        x = layers.Activation("sigmoid", name=f"{name}_sigmoid")(x)
    else:
        x = layers.Activation("softmax", name=f"{name}_softmax")(x)

    return x


# =========================================================
# Fast-SCNN Builder
# =========================================================
def build_fast_scnn(
    input_shape=(256, 256, 3),
    num_classes=4,
    down_channels=(32, 48, 64),
    block_channels=(64, 96, 128),
    bottleneck_counts=(3, 3, 3),
    expansion=6,
    ppm_channels=128,
    fusion_channels=128,
    classifier_channels=128,
):
    """
    Build Fast-SCNN for multiclass segmentation.
    """
    inputs = layers.Input(shape=input_shape, name="input_image")

    high_res = learning_to_downsample(inputs, down_channels=down_channels, name="ltd")
    low_res = global_feature_extractor(
        high_res,
        block_channels=block_channels,
        bottleneck_counts=bottleneck_counts,
        expansion=expansion,
        ppm_channels=ppm_channels,
        name="gfe"
    )
    fused = feature_fusion_module(high_res, low_res, out_channels=fusion_channels, name="ffm")
    outputs = classifier_head(
        fused,
        num_classes=num_classes,
        classifier_channels=classifier_channels,
        dropout_rate=0.1,
        name="classifier"
    )

    model = Model(inputs=inputs, outputs=outputs, name="FastSCNN")
    return model


# =========================================================
# Parameter Counter
# =========================================================
def count_model_parameters(model):
    trainable_params = int(
        tf.reduce_sum([tf.reduce_prod(v.shape) for v in model.trainable_variables])
    )
    non_trainable_params = int(
        tf.reduce_sum([tf.reduce_prod(v.shape) for v in model.non_trainable_variables])
    )
    total_params = trainable_params + non_trainable_params
    return total_params, trainable_params, non_trainable_params


# =========================================================
# Optional Loss
# =========================================================
def multiclass_dice_loss(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
    denominator = tf.reduce_sum(y_true + y_pred, axis=[1, 2])

    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return 1.0 - tf.reduce_mean(dice)


def combined_ce_dice_loss(y_true, y_pred):
    ce = tf.keras.losses.CategoricalCrossentropy()(y_true, y_pred)
    dl = multiclass_dice_loss(y_true, y_pred)
    return ce + dl


# =========================================================
# Main
# =========================================================
def main():
    INPUT_SHAPE = (256, 256, 3)
    NUM_CLASSES = 3

    model = build_fast_scnn(
        input_shape=INPUT_SHAPE,
        num_classes=NUM_CLASSES,

        # Default setting
        down_channels=(32, 48, 64),
        block_channels=(64, 96, 128),
        bottleneck_counts=(2, 2, 2),
        expansion=4,
        ppm_channels=64,
        fusion_channels=64,
        classifier_channels=64,
    )

    model.summary()

    total_params, trainable_params, non_trainable_params = count_model_parameters(model)

    print("\nParameter counts:")
    print(f"Total params:         {total_params:,}")
    print(f"Trainable params:     {trainable_params:,}")
    print(f"Non-trainable params: {non_trainable_params:,}")

    x = tf.random.normal((2, *INPUT_SHAPE))
    y = model(x)

    print("\nDummy forward pass:")
    print(f"Input shape : {x.shape}")
    print(f"Output shape: {y.shape}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=combined_ce_dice_loss,
        metrics=["accuracy"]
    )

    print("\nModel compiled successfully.")


if __name__ == "__main__":
    main()