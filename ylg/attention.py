import masks as sparse
import tensorflow as tf
from ops import sn_conv1x1


def get_grid_masks(gridO, gridI, nH=8, filling_curve='manhattan'):
    rtl = [
        sparse.RightFloorMask.get_grid_mask_from_1d(gridI, nO=gridO, filling_curve=filling_curve),
        sparse.RightRepetitiveMask.get_grid_mask_from_1d(gridI, nO=gridO, filling_curve=filling_curve)
    ]

    ltr = [
        sparse.LeftFloorMask.get_grid_mask_from_1d(gridI, nO=gridO, filling_curve=filling_curve),
        sparse.LeftRepetitiveMask.get_grid_mask_from_1d(gridI, nO=gridO, filling_curve=filling_curve)
    ]

    masks = rtl + ltr

    if nH == 8:
        masks = [mask for mask in masks for _ in (0, 1)]

    return np.array(masks)


def sn_attention_block_sim(x, training=True, nH=4, name='sn_nonlocal'):
    """
        "Sparse" Attention block **with** downsampling + learnable residual
        connection.
    """
    with tf.compat.v1.variable_scope(name):
        _, h, w, num_channels = x.shape.as_list()
        head_size = (num_channels // 8) // nH

        masks = get_grid_masks((h, w), (h // 2, w // 2), nH=nH)

        query = sn_conv1x1(x, num_channels // 8, training, 'sn_conv_theta')
        query = tf.reshape(query, [-1, h * w, nH, head_size])
        query = tf.transpose(query, [0, 2, 1, 3])

        key = sn_conv1x1(x, num_channels // 8, training, 'sn_conv_phi')
        key = tf.compat.v1.layers.max_pooling2d(inputs=key, pool_size=[2, 2],
                                                strides=2)
        key = tf.reshape(key, [-1, h * w // 4, nH, head_size])
        key = tf.transpose(key, [0, 2, 1, 3])

        # calculate attention map
        attn = tf.matmul(query, key, transpose_b=True)
        adder = (1.0 - masks) * (-1000.0)
        attn += adder
        attn = tf.nn.softmax(attn)

        v_head_size = (num_channels // 2) // nH
        value = sn_conv1x1(x, num_channels // 2, training, 'sn_conv_g')
        value = tf.compat.v1.layers.max_pooling2d(
            inputs=value, pool_size=[2, 2], strides=2)
        value = tf.reshape(value, [-1, h * w // 4, nH, v_head_size])
        value = tf.transpose(value, [0, 2, 1, 3])

        # calculate attention value
        attn_value = tf.matmul(attn, value)
        attn_value = tf.transpose(attn_value, [0, 2, 3, 1])
        attn_value = tf.reshape(attn_value, [-1, h, w, num_channels // 2])

        # Convolutional transform of attention output
        attn_value = sn_conv1x1(attn_value, num_channels, training, 'sn_conv_attn')

        # Learnable residual
        sigma = tf.compat.v1.get_variable(
            'sigma_ratio', [], initializer=tf.compat.v1.initializers.constant(0.0))
        return x + sigma * value


def sn_non_local_block_sim(x, training=True, name='sn_nonlocal',
                           nH=1):
    """
        Attention block **with** downsampling + learnable residual
        connection.
    """
    with tf.compat.v1.variable_scope(name):
        _, h, w, num_channels = x.shape.as_list()
        head_size = (num_channels // 8) // nH

        query = sn_conv1x1(x, num_channels // 8, training, 'sn_conv_theta')
        query = tf.reshape(query, [-1, h * w, nH, head_size])
        query = tf.transpose(query, [0, 2, 1, 3])

        key = sn_conv1x1(x, num_channels // 8, training, 'sn_conv_phi')
        key = tf.compat.v1.layers.max_pooling2d(inputs=key, pool_size=[2, 2],
                                                strides=2)
        key = tf.reshape(key, [-1, h * w // 4, nH, head_size])
        key = tf.transpose(key, [0, 2, 1, 3])

        # calculate attention map
        attn = tf.matmul(query, key, transpose_b=True)
        attn = tf.nn.softmax(attn)

        v_head_size = (num_channels // 2) // nH
        value = sn_conv1x1(x, num_channels // 2, training, 'sn_conv_g')
        value = tf.compat.v1.layers.max_pooling2d(
            inputs=value, pool_size=[2, 2], strides=2)
        value = tf.reshape(value, [-1, h * w // 4, nH, v_head_size])
        value = tf.transpose(value, [0, 2, 1, 3])

        # calculate attention value
        attn_value = tf.matmul(attn, value)
        attn_value = tf.transpose(attn_value, [0, 2, 3, 1])
        attn_value = tf.reshape(attn_value, [-1, h, w, num_channels // 2])

        # Convolutional transform of attention output
        attn_value = sn_conv1x1(attn_value, num_channels, training, 'sn_conv_attn')

        # Learnable residual
        sigma = tf.compat.v1.get_variable(
            'sigma_ratio', [], initializer=tf.compat.v1.initializers.constant(0.0))
        return x + sigma * attn_value
