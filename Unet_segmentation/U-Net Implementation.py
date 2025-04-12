from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv3D, MaxPooling3D, concatenate, Conv3DTranspose,
    BatchNormalization, Dropout, Lambda, Activation, SpatialDropout3D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K

def advanced_unet_model(
    input_shape, 
    num_classes,
    filters=16,
    dropout_rate=0.1,
    batch_norm=True,
    activation='relu',
    deep_supervision=False,
    l2_reg=1e-5
):
    """
    Advanced 3D U-Net model with several improvements over the basic version.
    
    Args:
        input_shape: Tuple of (height, width, depth, channels)
        num_classes: Number of output classes
        filters: Number of base filters (default 16)
        dropout_rate: Dropout rate (default 0.1)
        batch_norm: Whether to use batch normalization (default True)
        activation: Activation function (default 'relu')
        deep_supervision: Whether to add deep supervision (default False)
        l2_reg: L2 regularization factor (default 1e-5)
        
    Returns:
        A compiled Keras model
    """
    inputs = Input(input_shape)
    
    # Helper function to create a convolutional block
    def conv_block(x, filters, kernel_size=(3, 3, 3), padding='same', strides=1):
        x = Conv3D(
            filters, kernel_size, 
            padding=padding, 
            strides=strides,
            activation=activation,
            kernel_initializer='he_normal',
            kernel_regularizer=l2(l2_reg)
        if batch_norm:
            x = BatchNormalization()(x)
        return x
    
    # Contracting Path
    c1 = conv_block(inputs, filters)
    c1 = conv_block(c1, filters)
    if dropout_rate > 0:
        c1 = SpatialDropout3D(dropout_rate)(c1)  # SpatialDropout is more effective for 3D
    p1 = MaxPooling3D((2, 2, 2))(c1)
    
    c2 = conv_block(p1, filters*2)
    c2 = conv_block(c2, filters*2)
    if dropout_rate > 0:
        c2 = SpatialDropout3D(dropout_rate)(c2)
    p2 = MaxPooling3D((2, 2, 2))(c2)
    
    c3 = conv_block(p2, filters*4)
    c3 = conv_block(c3, filters*4)
    if dropout_rate > 0:
        c3 = SpatialDropout3D(dropout_rate*1.5)(c3)  # Slightly higher dropout in deeper layers
    p3 = MaxPooling3D((2, 2, 2))(c3)
    
    c4 = conv_block(p3, filters*8)
    c4 = conv_block(c4, filters*8)
    if dropout_rate > 0:
        c4 = SpatialDropout3D(dropout_rate*1.5)(c4)
    p4 = MaxPooling3D((2, 2, 2))(c4)
    
    # Bottleneck
    c5 = conv_block(p4, filters*16)
    c5 = conv_block(c5, filters*16)
    if dropout_rate > 0:
        c5 = SpatialDropout3D(dropout_rate*2)(c5)
    
    # Expanding Path
    u6 = Conv3DTranspose(filters*8, (2, 2, 2), strides=(2, 2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = conv_block(u6, filters*8)
    c6 = conv_block(c6, filters*8)
    
    u7 = Conv3DTranspose(filters*4, (2, 2, 2), strides=(2, 2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = conv_block(u7, filters*4)
    c7 = conv_block(c7, filters*4)
    
    u8 = Conv3DTranspose(filters*2, (2, 2, 2), strides=(2, 2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = conv_block(u8, filters*2)
    c8 = conv_block(c8, filters*2)
    
    u9 = Conv3DTranspose(filters, (2, 2, 2), strides=(2, 2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    c9 = conv_block(u9, filters)
    c9 = conv_block(c9, filters)
    
    # Output
    outputs = Conv3D(
        num_classes, (1, 1, 1), 
        activation='softmax', 
        kernel_initializer='glorot_uniform')(c9)
    
    # Optional Deep Supervision
    if deep_supervision:
        # Add intermediate outputs for deep supervision
        o1 = Conv3D(num_classes, (1, 1, 1), activation='softmax')(c6)
        o1 = Conv3DTranspose(num_classes, (4, 4, 4), strides=(4, 4, 4), padding='same')(o1)
        
        o2 = Conv3D(num_classes, (1, 1, 1), activation='softmax')(c7)
        o2 = Conv3DTranspose(num_classes, (2, 2, 2), strides=(2, 2, 2), padding='same')(o2)
        
        outputs = [outputs, o2, o1]
    
    model = Model(inputs=[inputs], outputs=[outputs])
    
    return model


def dice_coefficient(y_true, y_pred, smooth=1.0):
    """
    Dice coefficient for 3D segmentation.
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    """
    Dice loss for optimization.
    """
    return 1 - dice_coefficient(y_true, y_pred)


def compile_model(model, learning_rate=1e-4, loss='dice_loss', metrics=None):
    """
    Compile the model with appropriate settings.
    """
    if metrics is None:
        metrics = [dice_coefficient, 'accuracy', MeanIoU(num_classes=4)]
    
    if loss == 'dice_loss':
        loss = dice_loss
    elif loss == 'categorical_crossentropy':
        loss = 'categorical_crossentropy'
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=metrics
    )
    
    return model


# Example usage
if __name__ == "__main__":
    # Define input shape and number of classes
    input_shape = (128, 128, 128, 3)  # height, width, depth, channels
    num_classes = 4
    
    # Create model
    model = advanced_unet_model(
        input_shape=input_shape,
        num_classes=num_classes,
        filters=16,
        dropout_rate=0.1,
        batch_norm=True,
        activation='relu',
        deep_supervision=False
    )
    
    # Compile model
    model = compile_model(model, learning_rate=1e-4)
    
    # Print model summary
    model.summary()
    
    print("Input shape:", model.input_shape)
    print("Output shape:", model.output_shape)
