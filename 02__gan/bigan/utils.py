import yaml, logging
import numpy as np

from tensorflow.python.keras.layers import (
    Input, Layer, Activation, Concatenate, Cropping2D, Add, BatchNormalization
)
import tensorflow as tf
from tensorflow.python.keras.layers.convolutional import UpSampling2D, Conv2D
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU

################################################################################

def deconv_output_length(input_length, filter_size, padding, stride):
  """Determines output length of a transposed convolution given input length.
  Arguments:
      input_length: integer.
      filter_size: integer.
      padding: one of "same", "valid", "full".
      stride: integer.
  Returns:
      The output length (integer).
  """
  if input_length is None:
    return None
  input_length *= stride
  if padding == 'valid':
    input_length += max(filter_size - stride, 0)
  elif padding == 'full':
    input_length -= (stride + filter_size - 2)
  return input_length


_allowed_reflections = {'off': ['off'],
                        'x_mirror': ['off', 'x'],
                        'y_mirror': ['off', 'y'],
                        'xy_mirror': ['off', 'x', 'xy', 'y'],
                        'd11_mirror': ['d11', 'off'],
                        'd1neg1_mirror': ['d1neg1', 'off'],
                        '45_mirror': ['d11', 'd1neg1', 'off', 'xy'],
                        '180_rotation': ['off', 'xy'],
                        '90_rotation': ['off', 'r90', 'rneg90', 'xy'],
                        'all': ['d11', 'd1neg1', 'off', 'r90', 'rneg90', 'x', 'xy', 'y']
}

def rot90_kernel(w, num_rots):
    [height,width,in_channels,out_channels] = w.shape

    for r in range(0, num_rots):
        w = tf.keras.backend.permute_dimensions(w, [1,0,2,3])
        w = tf.keras.backend.reverse(w,[ 0])

    return w

def mirror_kernel_2(w, axes):

    [height,width,in_channels,out_channels] = w.shape
    for axis in axes:
        #dimen = height if axis==1 else width
        kernels = tf.keras.backend.reverse(w, [ axis])
        w = kernels
    return w



class SymmConv2D(Conv2D):
  def __init__(self, symmetry, symmetry_comb="add", input_is_concat=False, kernel_set_weights = [None, None], *args, **kwargs):
    self.symmetry = 'off' if not symmetry else symmetry
    self.symmetry_comb = symmetry_comb
    self.input_is_concat = input_is_concat
    self.symm_factor = len(_allowed_reflections[self.symmetry] )
    self._kernel_set_weights = kernel_set_weights
    self._kernel = None
    self._bias = None
    super(SymmConv2D, self).__init__(*args,**kwargs)

  def get_config(self):
    config = {
        'symmetry': self.symmetry,
        'symmetry_comb': self.symmetry_comb,
        'input_is_concat':self.input_is_concat,
    }
    base_config = super(SymmConv2D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


  def build(self, input_shape):
    if self.data_format == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1
    if input_shape[channel_axis] is None:
        raise ValueError('The channel dimension of the inputs '
                         'should be defined. Found `None`.')
    input_dim = input_shape[channel_axis]

    if self.symmetry_comb == "concatenate":
      if self.filters%self.symm_factor != 0:
        raise ValueError("filters must be a multiple of symm_factor when symmetry_comb=concatenate")
      orig_filters_size = int(self.filters // self.symm_factor)
      if self.input_is_concat:
        orig_kernel_input_size = int(input_dim // self.symm_factor)
      else:
        orig_kernel_input_size =  input_dim

    #elif self.symmetry_comb == "add":
    else:
      orig_filters_size =  self.filters
      orig_kernel_input_size =  input_dim
      #if input_dim%self.symm_factor != 0:
      #  raise ValueError("input_dim must be a multiple of symm_factor when symmetry_comb=add")
      #orig_kernel_input_size = int(input_dim // self.symm_factor)

    kernel_shape = self.kernel_size + (orig_kernel_input_size, orig_filters_size )

    self._kernel  = self.add_weight(shape=kernel_shape,
                                  initializer=self.kernel_initializer,
                                  name='kernel',
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
    if self._kernel_set_weights[0] is not None:
      self.kernel_raw = self._kernel_set_weights[0]
    self._update_kernel_dict()

    if self.use_bias:
      self._bias = self.add_weight(shape=(orig_filters_size,),
                                   initializer=self.bias_initializer,
                                   name='bias',
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint)
      if self._kernel_set_weights[1] is not None:
        self.bias = self._kernel_set_weights[1]
    else:
        self._bias = None
    # Set input spec.
    self.input_spec = tf.keras.layers.InputSpec(ndim=self.rank + 2, axes={channel_axis: input_dim})
    self.built = True

  def _update_kernel_dict(self):
    self._kernel_keys_list = []
    self._kernel_dict = {}
    self._kernel_dict["off"] = self._kernel
    self._kernel_keys_list.append("off")
    for symm_op in _allowed_reflections[self.symmetry]:
      if symm_op != "off":
        self._kernel_keys_list.append(symm_op)
      if symm_op == "x":
        self._kernel_dict[symm_op] = mirror_kernel_2(self._kernel, [0,])
      elif symm_op == "y":
        self._kernel_dict[symm_op] = mirror_kernel_2(self._kernel, [1,])
      elif symm_op == "xy":
        self._kernel_dict[symm_op] = mirror_kernel_2(self._kernel, [0,1])
      elif symm_op == "r90":
        self._kernel_dict[symm_op] = rot90_kernel(self._kernel, 1)
      elif symm_op == "rneg90":
        self._kernel_dict[symm_op] = rot90_kernel(self._kernel, 3)
      elif symm_op == "d11":
        self._kernel_dict[symm_op] = tf.keras.backend.permute_dimensions(self._kernel, [1,0,2,3])
      elif symm_op == "d1neg1":
        self._kernel_dict[symm_op] =  mirror_kernel_2(tf.keras.backend.permute_dimensions(self._kernel, [1,0,2,3])  , [0,1])

  @property
  def bias(self):
    if self._bias is None:
      return None
    else:
      return tf.keras.backend.get_value(self._bias)

  @bias.setter
  def bias(self, b):
    if self.use_bias:
      tf.keras.backend.set_value(self._bias, b)

  @property
  def kernel_raw(self):
    # returns a numpy array of the kernel ("off", before symmetrization)
    if self._kernel is None:
      return None
    return tf.keras.backend.get_value(self._kernel)

  @kernel_raw.setter
  def kernel_raw(self, w):
    # w is a numpy array of kernel ("off", before symmetrization)
    if self._kernel is not None:
      tf.keras.backend.set_value(self._kernel, w)
      self._update_kernel_dict()

  @property
  def kernel(self):
    try:
      total_weight = 0
      cnt = 0
      for tensor in self.kernel_dict.values():
        total_weight += tf.keras.backend.get_value(tensor)
        cnt += 1
      return total_weight/cnt
    except AttributeError:
      return None

  @property
  def kernel_dict(self):
    return self._kernel_dict

  def call(self, inputs_orig):
    inputs = inputs_orig
    total_kernel = None
    # if is_multi_cpu:
    #   self._update_kernel_dict() # in eager mode, needs to update kernel explictly
    self._update_kernel_dict() # in eager mode, needs to update kernel explictly
    for kernel_key in self._kernel_keys_list:
      if total_kernel is None:
        total_kernel = self._kernel_dict[kernel_key]
        num_kernels = 1
      else:
        total_kernel = total_kernel + self._kernel_dict[kernel_key]
        num_kernels += 1

    total_kernel = total_kernel / num_kernels

    outputs = tf.keras.backend.conv2d(
          inputs,
          total_kernel,
          strides=self.strides,
          padding=self.padding,
          data_format=self.data_format,
          dilation_rate=self.dilation_rate)

    if self.use_bias:
      outputs = tf.keras.backend.bias_add(outputs, self._bias, data_format=self.data_format)

    if self.activation is not None:
    #   print("self.activation: ", self.activation)
      outputs = self.activation(outputs)

    if self.symmetry_comb == "concatenate":
      logger.error(" - symmetry_comb 'concatenate' is not supported. Use 'add' only.")
      ##now stack the outputs
      #totals = []
      #for outputs in outputs_per_input: # TODO syntax error, no such outputs_per_input
      #  #add within this entry
      #  total=None
      #  for op in outputs:
      #    if total is None:
      #      total = op
      #    else:
      #      total = total + op
      #  totals.append(total)

      #final_outputs = tf.keras.backend.concatenate(totals,axis=3  )

    #elif self.symmetry_comb == "add":
    else:
      final_outputs = outputs
    return final_outputs

################################################################################

class SkipConv(Layer):
    def __init__(self, cropping, channel):
        # super().__init__()
        super(SkipConv, self).__init__()
        self.cropping = cropping
        self.channel = channel
        self.myCrop2D = Cropping2D(cropping=cropping)
        self.conv2D1x1 = Conv2D(filters=channel, kernel_size=1,
            activation=None)
        # self.conv2D1x1 = Conv2D(filters=channel, kernel_size=1,
        #     use_bias=False, activation=None)
 
    def call(self, inputs):
        input_crop = self.myCrop2D(inputs)
        if inputs.shape[-1] != self.channel:
            # print("<conv2D1x1> inputs.shape[-1]: ", inputs.shape[-1], "  self.channel: ", self.channel)
            output = self.conv2D1x1(input_crop)
        else:
            output = input_crop
        return output 
 
def build_model(in_channels, out_channels, kernels, channels, paddings, symmetries, activations, batchNorms, cba, skipConnections, skipType="concat", dilations=None, reg_l1=0.0, reg_l2=0.0, target=False):
    assert len(kernels)==len(channels)
    assert len(kernels)==len(skipConnections)
    assert len(kernels)==len(symmetries)
    assert len(kernels)==len(paddings)
    assert len(kernels)==len(activations)
    assert len(kernels)==len(batchNorms)
    if dilations is None:
        dilations = [1]*len(kernels)
    
    effective_kernels = [k+(k-1)*(d-1) for (k, d) in zip(kernels, dilations)]
    layerID = range(len(kernels))

    #
    if (reg_l1 == 0.0) and (reg_l2 == 0.0): 
        kernel_regularizer = None
    else:
        kernel_regularizer = tf.keras.regularizers.l1_l2(l1=reg_l1, l2=reg_l2)
    print("regularizer: ", kernel_regularizer)
    #

    def get_actf(actf):
        if actf=="relu": return Activation('relu')
        elif actf=="leakyrelu": return LeakyReLU(alpha=0.1)
        # elif actf=="leakyrelu": return LeakyReLU(alpha=0.2)
        elif actf=="sigmoid": return Activation('sigmoid')
        elif actf=="softplus": return Activation('softplus')
        elif actf=="tanh": return Activation('tanh')
        # elif actf=="linear": return Activation('linear')
        elif actf=="linear": return None
 
    img_shape = (None, None, in_channels)
    input_image = Input(shape=img_shape)
    if target:  # for discriminator
        source_image = Input(shape=img_shape)
        x = Concatenate(axis=-1)([source_image, input_image])
    else:
        x = input_image
    
    outputs=[x]
    for i, (ks, channel, dilation, padding, symmetry, skipCon, activation, batchNorm) in enumerate(zip(kernels, channels, dilations, paddings, symmetries, skipConnections, activations, batchNorms)):
        _conv = SymmConv2D(symmetry=symmetry, filters=channel, kernel_size=ks, strides=1, dilation_rate=dilation, padding=padding, kernel_regularizer=kernel_regularizer)
        y = _conv(outputs[-1])
        
        if batchNorm and cba: y = BatchNormalization(momentum=0.8)(y)
        actf = get_actf(activation)
        if actf is not None: y = actf(y)
        if batchNorm and (not cba): y = BatchNormalization(momentum=0.8)(y)
        outputs.append(y)

        # handle the skips
        if skipCon == -1:
            continue
        else:
            assert skipCon >= 0
            inp_layerID = skipCon
            out_layerID = i+1
            inp_tensor = outputs[inp_layerID]
            out_tensor = outputs[out_layerID]
            #
            cropping = sum(effective_kernels[inp_layerID:out_layerID])-(out_layerID-inp_layerID)
            cropping = cropping // 2
            if skipType == "add":
                # inp_tensor_channel = 1 if skipCon == 0 else channels[skipCon-1]
                inp_tensor_crop = SkipConv(cropping, channels[i])(inp_tensor)
                out_tensor_new = Add()([out_tensor, inp_tensor_crop])
            else: # skipType == "concat"
                inp_tensor_crop = Cropping2D(cropping)(inp_tensor)
                out_tensor_new = Concatenate()([out_tensor, inp_tensor_crop])

            outputs[i+1] = out_tensor_new
    #
    model_ambit = sum(effective_kernels) - len(kernels)
    #
    if target:
        return Model([source_image, input_image], outputs[-1]), model_ambit
    else:
        return Model(inputs=input_image, outputs=outputs[-1]), model_ambit


def init_logging(filename, checkpoint=None):
    # mode = "a+" if checkpoint is not None else "w+"
    mode = "w+"
    logging.basicConfig(
        level=logging.INFO,
        # format="%(asctime)s [%(levelname)s] %(message)s",
        format="%(message)s",
        handlers=[logging.FileHandler(filename, mode=mode), logging.StreamHandler()],
    )

def read_config(config):
    with open(config, 'r') as f:
        config_dict = yaml.safe_load(f)
    return config_dict


def transfer_weight(from_model, to_model):
    from_lls, to_lls = from_model.layers, to_model.layers
    from_i, to_i = 0, 0
    num_layers = min(len(from_lls), len(to_lls))
    print("from_lls: {}, to_lls: {}, num_layers: {}".format(len(from_lls),len(to_lls),num_layers))

    for from_l in from_lls:
        if isinstance(from_l, Cropping2D): continue
        to_l = to_lls[to_i]
        to_i += 1
        if isinstance(to_l, Cropping2D):
            to_l = to_lls[to_i]
            to_i += 1

        if isinstance(from_l, Conv2D):
            if not isinstance(to_l, Conv2D): print("!!!!!!!!!! MISMATCH !!!!!!!!!!")
            else: to_l.set_weights(from_l.get_weights())
        elif isinstance(from_l, BatchNormalization):
            if not isinstance(to_l, BatchNormalization): print("!!!!!!!!!! MISMATCH !!!!!!!!!!")
            else: to_l.set_weights(from_l.get_weights())

        # else:
        #     print(from_l)

    # print("FROM_MODEL")
    # for from_l in from_lls:
    #     print(from_l)
    # print("TO_MODEL")
    # for to_l in to_lls:
    #     print(to_l)

if __name__ == "__main__":
    print( read_config("training_config.yaml") )

