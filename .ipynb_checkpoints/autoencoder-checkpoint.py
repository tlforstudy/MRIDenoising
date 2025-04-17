from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Concatenate, BatchNormalization, LeakyReLU, ReLU,UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

def get_autoencoder_model(img_width=64, img_height=64):

    return None

from tensorflow.keras.layers import Layer, Conv2D, Dense, GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.layers import Reshape, Multiply, Add, Activation, Concatenate



class ChannelAttention(Layer):
    def __init__(self, channels, reduction_ratio=8, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.channels = channels
        self.reduction_ratio = reduction_ratio
        self.mlp = tf.keras.Sequential([
            Dense(channels // reduction_ratio, activation='relu'),
            Dense(channels)
        ])
        
    def build(self, input_shape):
        self.avg_pool = GlobalAveragePooling2D()
        self.max_pool = GlobalMaxPooling2D()
        
    def call(self, inputs):
        avg_out = self.mlp(self.avg_pool(inputs))
        max_out = self.mlp(self.max_pool(inputs))
        
        channel_weights = tf.nn.sigmoid(avg_out + max_out)
        channel_weights = Reshape((1, 1, self.channels))(channel_weights)
        
        return Multiply()([inputs, channel_weights])
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "channels": self.channels,
            "reduction_ratio": self.reduction_ratio
        })
        return config

class SpatialAttention(Layer):
    def __init__(self, kernel_size=7, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.conv = Conv2D(1, kernel_size, padding='same', activation='sigmoid')
        
    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        
        concat = Concatenate()([avg_pool, max_pool])
        spatial_weights = self.conv(concat)
        
        return Multiply()([inputs, spatial_weights])
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "kernel_size": self.kernel_size
        })
        return config

class CBAM(Layer):
    def __init__(self, channels, reduction_ratio=8, kernel_size=7, **kwargs):

        super(CBAM, self).__init__(**kwargs)
        self.channels = channels
        self.reduction_ratio = reduction_ratio
        
        self.kernel_size = kernel_size
        self.channel_att = ChannelAttention(channels, reduction_ratio)
        self.spatial_att = SpatialAttention(kernel_size)

        
        
    def call(self, inputs):
        x = self.channel_att(inputs)
        x = self.spatial_att(x)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "channels": self.channels,
            "reduction_ratio": self.reduction_ratio,
            "kernel_size": self.kernel_size,
        })
        return config
    
def edge_loss(y_true,y_pred,max_val=1.0):
    y_true = y_true/max_val
    y_pred = y_pred/max_val
    edge_true = tf.image.sobel_edges(y_true)
    edge_pred = tf.image.sobel_edges(y_pred)
    edge_true = tf.sqrt(tf.reduce_sum(tf.square(edge_true), axis=-1) + 1e-8)
    edge_pred = tf.sqrt(tf.reduce_sum(tf.square(edge_pred), axis=-1) + 1e-8)
    current_max = tf.reduce_max(tf.maximum(edge_true, edge_pred))
    edge_true = edge_true / current_max
    edge_pred = edge_pred / current_max

    return tf.reduce_mean(tf.square(edge_true - edge_pred))

def ssim(y_true, y_pred):
    return tf.image.ssim(y_true, y_pred, max_val=1.0)

def mixed_loss(y_true, y_pred, alpha=0.15,beta=0.05,gamma=0.1,max_val=1.0):
    mse_loss = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
    ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=max_val))
    edgeloss = edge_loss(y_true,y_pred,max_val=1.0)
    l1_loss = tf.keras.losses.MeanAbsoluteError()(y_true, y_pred)
    
    return (1-alpha-beta-gamma) * mse_loss + alpha * ssim_loss + beta * edgeloss+ gamma*l1_loss

def mix_loss(y_true, y_pred):
    return mixed_loss(y_true, y_pred, alpha=0.15,beta=0.05,gamma=0.1, max_val=1.0)
    #return mixed_loss(y_true, y_pred, alpha=0.2,beta=0.1,gamma=0.5, max_val=1.0)


def get_autoencoder_model_upsample(img_width=256, img_height=256):
    inputs = Input(shape=(img_width, img_height, 1))
    
    # Encoder
    conv1 = Conv2D(64, (5, 5), padding='same')(inputs)
    conv1 = LeakyReLU(alpha=0.1)(conv1)
    conv1 = CBAM(64)(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    
    conv2 = Conv2D(128, (3, 3), padding='same')(pool1)
    conv2 = LeakyReLU(alpha=0.1)(conv2)
    conv2 = CBAM(128)(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    
    conv3 = Conv2D(256, (3, 3), padding='same')(pool2)
    conv3 = LeakyReLU(alpha=0.1)(conv3)
    conv3 = CBAM(256)(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    
    
    # Bottleneck
    conv4 = Conv2D(512, (3, 3), padding='same')(pool3)
    conv4 = LeakyReLU(alpha=0.1)(conv4)
    conv4 = CBAM(512)(conv4)
    conv4 = BatchNormalization()(conv4)

    

    
    # Decoder
    #up5 = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(conv4)
    up5 = UpSampling2D(size=(2, 2), interpolation='bilinear')(conv4)
    up5 = Conv2D(256, (3, 3), padding='same')(up5)
    #cbam_conv3 = CBAM(256)(conv3)
    concat5 = Concatenate()([up5, conv3])
    conv5 = Conv2D(256, (3, 3), padding='same')(concat5)
    conv5 = LeakyReLU(alpha=0.1)(conv5)
    conv5 = BatchNormalization()(conv5)
    
    #up6 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(conv5)
    up6 = UpSampling2D(size=(2, 2), interpolation='bilinear')(conv5)
    up6 = Conv2D(128, (3, 3), padding='same')(up6)
    #cbam_conv2 = CBAM(128)(conv2)
    concat6 = Concatenate()([up6, conv2])
    conv6 = Conv2D(128, (3, 3), padding='same')(concat6)
    conv6 = LeakyReLU(alpha=0.1)(conv6)
    conv6 = BatchNormalization()(conv6)
    
    #up7 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(conv6)
    up7 = UpSampling2D(size=(2, 2), interpolation='bilinear')(conv6)
    up7 = Conv2D(64, (3, 3), padding='same')(up7)
    cbam_conv1 = CBAM(64)(conv1)
    concat7 = Concatenate()([up7, cbam_conv1])
    conv7 = Conv2D(64, (3, 3), padding='same')(concat7)
    conv7 = LeakyReLU(alpha=0.1)(conv7)
    conv7 = BatchNormalization()(conv7)
    
    # Output
    outputs = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(conv7)
    
    
    unet = Model(inputs=inputs, outputs=outputs)
    
    
    unet.compile(optimizer=Adam(learning_rate=1e-4), 
                 loss=mix_loss,
                 metrics=[tf.keras.losses.MeanAbsoluteError(),ssim,'mse'])
    return unet

def get_autoencoder_model_transpose(img_width=256, img_height=256):
    inputs = Input(shape=(img_width, img_height, 1))
    
    # Encoder
    conv1 = Conv2D(64, (5, 5), padding='same')(inputs)
    conv1 = LeakyReLU(alpha=0.1)(conv1)
    conv1 = CBAM(64)(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    
    conv2 = Conv2D(128, (3, 3), padding='same')(pool1)
    conv2 = LeakyReLU(alpha=0.1)(conv2)
    conv2 = CBAM(128)(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    
    conv3 = Conv2D(256, (3, 3), padding='same')(pool2)
    conv3 = LeakyReLU(alpha=0.1)(conv3)
    conv3 = CBAM(256)(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    
    
    # Bottleneck
    conv4 = Conv2D(512, (3, 3), padding='same')(pool3)
    conv4 = LeakyReLU(alpha=0.1)(conv4)
    conv4 = CBAM(512)(conv4)
    conv4 = BatchNormalization()(conv4)

    

    
    # Decoder
    up5 = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(conv4)
    #up5 = UpSampling2D(size=(2, 2), interpolation='bilinear')(conv4)
    #up5 = Conv2D(256, (3, 3), padding='same')(up5)
    concat5 = Concatenate()([up5, conv3])
    conv5 = Conv2D(256, (3, 3), padding='same')(concat5)
    conv5 = LeakyReLU(alpha=0.1)(conv5)
    conv5 = BatchNormalization()(conv5)
    
    up6 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(conv5)
    #up6 = UpSampling2D(size=(2, 2), interpolation='bilinear')(conv5)
    #up6 = Conv2D(128, (3, 3), padding='same')(up6)
    concat6 = Concatenate()([up6, conv2])
    conv6 = Conv2D(128, (3, 3), padding='same')(concat6)
    conv6 = LeakyReLU(alpha=0.1)(conv6)
    conv6 = BatchNormalization()(conv6)
    
    up7 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(conv6)
    #up7 = UpSampling2D(size=(2, 2), interpolation='bilinear')(conv6)
    #up7 = Conv2D(64, (3, 3), padding='same')(up7)
    cbam_conv1 = CBAM(64)(conv1)
    concat7 = Concatenate()([up7, cbam_conv1])
    conv7 = Conv2D(64, (3, 3), padding='same')(concat7)
    conv7 = LeakyReLU(alpha=0.1)(conv7)
    conv7 = BatchNormalization()(conv7)
    
    # Output
    outputs = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(conv7)
    
    
    unet = Model(inputs=inputs, outputs=outputs)
    
    
    unet.compile(optimizer=Adam(learning_rate=1e-4), 
                 loss=mix_loss,
                 metrics=[tf.keras.losses.MeanAbsoluteError(),ssim,'mse'])
    return unet


def get_autoencoder_model_NOCBAM(img_width=256, img_height=256):
    inputs = Input(shape=(img_width, img_height, 1))
    
    # Encoder
    conv1 = Conv2D(64, (5, 5), padding='same')(inputs)
    conv1 = LeakyReLU(alpha=0.1)(conv1)
    #conv1 = CBAM(64)(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    
    conv2 = Conv2D(128, (3, 3), padding='same')(pool1)
    conv2 = LeakyReLU(alpha=0.1)(conv2)
    #conv2 = CBAM(128)(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    
    conv3 = Conv2D(256, (3, 3), padding='same')(pool2)
    conv3 = LeakyReLU(alpha=0.1)(conv3)
    #conv3 = CBAM(256)(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    
    
    # Bottleneck
    conv4 = Conv2D(512, (3, 3), padding='same')(pool3)
    conv4 = LeakyReLU(alpha=0.1)(conv4)
    #conv4 = CBAM(512)(conv4)
    conv4 = BatchNormalization()(conv4)

    

    
    # Decoder
    #up5 = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(conv4)
    up5 = UpSampling2D(size=(2, 2), interpolation='bilinear')(conv4)
    up5 = Conv2D(256, (3, 3), padding='same')(up5)
    concat5 = Concatenate()([up5, conv3])
    conv5 = Conv2D(256, (3, 3), padding='same')(concat5)
    conv5 = LeakyReLU(alpha=0.1)(conv5)
    conv5 = BatchNormalization()(conv5)
    
    #up6 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(conv5)
    up6 = UpSampling2D(size=(2, 2), interpolation='bilinear')(conv5)
    up6 = Conv2D(128, (3, 3), padding='same')(up6)
    concat6 = Concatenate()([up6, conv2])
    conv6 = Conv2D(128, (3, 3), padding='same')(concat6)
    conv6 = LeakyReLU(alpha=0.1)(conv6)
    conv6 = BatchNormalization()(conv6)
    
    #up7 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(conv6)
    up7 = UpSampling2D(size=(2, 2), interpolation='bilinear')(conv6)
    up7 = Conv2D(64, (3, 3), padding='same')(up7)
    #cbam_conv1 = CBAM(64)(conv1)
    concat7 = Concatenate()([up7, conv1])
    conv7 = Conv2D(64, (3, 3), padding='same')(concat7)
    conv7 = LeakyReLU(alpha=0.1)(conv7)
    conv7 = BatchNormalization()(conv7)
    
    # Output
    outputs = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(conv7)
    
    
    unet = Model(inputs=inputs, outputs=outputs)
    
    
    unet.compile(optimizer=Adam(learning_rate=1e-4), 
                 loss=mix_loss,
                 metrics=[tf.keras.losses.MeanAbsoluteError(),ssim,'mse'])
    return unet

def get_autoencoder_model_noskipnoCBAM(img_width=256, img_height=256):
    inputs = Input(shape=(img_width, img_height, 1))
    
    # Encoder
    conv1 = Conv2D(64, (5, 5), padding='same')(inputs)
    conv1 = LeakyReLU(alpha=0.1)(conv1)
    #conv1 = CBAM(64)(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    
    conv2 = Conv2D(128, (3, 3), padding='same')(pool1)
    conv2 = LeakyReLU(alpha=0.1)(conv2)
    #conv2 = CBAM(128)(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    
    conv3 = Conv2D(256, (3, 3), padding='same')(pool2)
    conv3 = LeakyReLU(alpha=0.1)(conv3)
    #conv3 = CBAM(256)(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    
    
    # Bottleneck
    conv4 = Conv2D(512, (3, 3), padding='same')(pool3)
    conv4 = LeakyReLU(alpha=0.1)(conv4)
    #conv4 = CBAM(512)(conv4)
    conv4 = BatchNormalization()(conv4)

    

    
    # Decoder
    #up5 = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(conv4)
    up5 = UpSampling2D(size=(2, 2), interpolation='bilinear')(conv4)
    up5 = Conv2D(256, (3, 3), padding='same')(up5)
    #concat5 = Concatenate()([up5, conv3])
    conv5 = Conv2D(256, (3, 3), padding='same')(up5)
    conv5 = LeakyReLU(alpha=0.1)(conv5)
    conv5 = BatchNormalization()(conv5)
    
    #up6 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(conv5)
    up6 = UpSampling2D(size=(2, 2), interpolation='bilinear')(conv5)
    up6 = Conv2D(128, (3, 3), padding='same')(up6)
    #concat6 = Concatenate()([up6, conv2])
    conv6 = Conv2D(128, (3, 3), padding='same')(up6)
    conv6 = LeakyReLU(alpha=0.1)(conv6)
    conv6 = BatchNormalization()(conv6)
    
    #up7 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(conv6)
    up7 = UpSampling2D(size=(2, 2), interpolation='bilinear')(conv6)
    up7 = Conv2D(64, (3, 3), padding='same')(up7)
    #cbam_conv1 = CBAM(64)(conv1)
    #concat7 = Concatenate()([up7, cbam_conv1])
    conv7 = Conv2D(64, (3, 3), padding='same')(up7)
    conv7 = LeakyReLU(alpha=0.1)(conv7)
    conv7 = BatchNormalization()(conv7)
    
    # Output
    outputs = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(conv7)
    
    
    unet = Model(inputs=inputs, outputs=outputs)
    
    
    unet.compile(optimizer=Adam(learning_rate=1e-4), 
                 loss=mix_loss,
                 metrics=[tf.keras.losses.MeanAbsoluteError(),ssim,'mse'])
    return unet





def get_autoencoder_model_CBAMSKIP(img_width=256, img_height=256):
    inputs = Input(shape=(img_width, img_height, 1))
    
    # Encoder
    conv1 = Conv2D(64, (5, 5), padding='same')(inputs)
    conv1 = LeakyReLU(alpha=0.1)(conv1)
    #conv1 = CBAM(64)(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    
    conv2 = Conv2D(128, (3, 3), padding='same')(pool1)
    conv2 = LeakyReLU(alpha=0.1)(conv2)
    #conv2 = CBAM(128)(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    
    conv3 = Conv2D(256, (3, 3), padding='same')(pool2)
    conv3 = LeakyReLU(alpha=0.1)(conv3)
    #conv3 = CBAM(256)(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    
    
    # Bottleneck
    conv4 = Conv2D(512, (3, 3), padding='same')(pool3)
    conv4 = LeakyReLU(alpha=0.1)(conv4)
    conv4 = CBAM(512)(conv4)
    conv4 = BatchNormalization()(conv4)

    

    
    # Decoder
    #up5 = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(conv4)
    up5 = UpSampling2D(size=(2, 2), interpolation='bilinear')(conv4)
    up5 = Conv2D(256, (3, 3), padding='same')(up5)
    cbam_conv3 = CBAM(256)(conv3)
    concat5 = Concatenate()([up5, cbam_conv3])
    conv5 = Conv2D(256, (3, 3), padding='same')(concat5)
    conv5 = LeakyReLU(alpha=0.1)(conv5)
    conv5 = BatchNormalization()(conv5)
    
    #up6 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(conv5)
    up6 = UpSampling2D(size=(2, 2), interpolation='bilinear')(conv5)
    up6 = Conv2D(128, (3, 3), padding='same')(up6)
    cbam_conv2 = CBAM(128)(conv2)
    concat6 = Concatenate()([up6, cbam_conv2])
    conv6 = Conv2D(128, (3, 3), padding='same')(concat6)
    conv6 = LeakyReLU(alpha=0.1)(conv6)
    conv6 = BatchNormalization()(conv6)
    
    #up7 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(conv6)
    up7 = UpSampling2D(size=(2, 2), interpolation='bilinear')(conv6)
    up7 = Conv2D(64, (3, 3), padding='same')(up7)
    cbam_conv1 = CBAM(64)(conv1)
    concat7 = Concatenate()([up7, cbam_conv1])
    conv7 = Conv2D(64, (3, 3), padding='same')(concat7)
    conv7 = LeakyReLU(alpha=0.1)(conv7)
    conv7 = BatchNormalization()(conv7)
    
    # Output
    outputs = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(conv7)
    
    
    unet = Model(inputs=inputs, outputs=outputs)
    
    
    unet.compile(optimizer=Adam(learning_rate=1e-4), 
                 loss=mix_loss,
                 metrics=[tf.keras.losses.MeanAbsoluteError(),ssim,'mse'])
    return unet




def get_autoencoder_model_CBAMencoder(img_width=256, img_height=256):
    inputs = Input(shape=(img_width, img_height, 1))
    
    # Encoder
    conv1 = Conv2D(64, (5, 5), padding='same')(inputs)
    conv1 = LeakyReLU(alpha=0.1)(conv1)
    conv1 = CBAM(64)(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    
    conv2 = Conv2D(128, (3, 3), padding='same')(pool1)
    conv2 = LeakyReLU(alpha=0.1)(conv2)
    conv2 = CBAM(128)(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    
    conv3 = Conv2D(256, (3, 3), padding='same')(pool2)
    conv3 = LeakyReLU(alpha=0.1)(conv3)
    conv3 = CBAM(256)(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    
    
    # Bottleneck
    conv4 = Conv2D(512, (3, 3), padding='same')(pool3)
    conv4 = LeakyReLU(alpha=0.1)(conv4)
    conv4 = CBAM(512)(conv4)
    conv4 = BatchNormalization()(conv4)

    

    
    # Decoder
    #up5 = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(conv4)
    up5 = UpSampling2D(size=(2, 2), interpolation='bilinear')(conv4)
    up5 = Conv2D(256, (3, 3), padding='same')(up5)
    #cbam_conv3 = CBAM(256)(conv3)
    concat5 = Concatenate()([up5, conv3])
    conv5 = Conv2D(256, (3, 3), padding='same')(concat5)
    conv5 = LeakyReLU(alpha=0.1)(conv5)
    conv5 = BatchNormalization()(conv5)
    
    #up6 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(conv5)
    up6 = UpSampling2D(size=(2, 2), interpolation='bilinear')(conv5)
    up6 = Conv2D(128, (3, 3), padding='same')(up6)
    #cbam_conv2 = CBAM(128)(conv2)
    concat6 = Concatenate()([up6, conv2])
    conv6 = Conv2D(128, (3, 3), padding='same')(concat6)
    conv6 = LeakyReLU(alpha=0.1)(conv6)
    conv6 = BatchNormalization()(conv6)
    
    #up7 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(conv6)
    up7 = UpSampling2D(size=(2, 2), interpolation='bilinear')(conv6)
    up7 = Conv2D(64, (3, 3), padding='same')(up7)
    #cbam_conv1 = CBAM(64)(conv1)
    concat7 = Concatenate()([up7, conv1])
    conv7 = Conv2D(64, (3, 3), padding='same')(concat7)
    conv7 = LeakyReLU(alpha=0.1)(conv7)
    conv7 = BatchNormalization()(conv7)
    
    # Output
    outputs = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(conv7)
    
    
    unet = Model(inputs=inputs, outputs=outputs)
    
    
    unet.compile(optimizer=Adam(learning_rate=1e-4), 
                 loss=mix_loss,
                 metrics=[tf.keras.losses.MeanAbsoluteError(),ssim,'mse'])
    return unet
