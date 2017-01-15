import mxnet as mx
from mxnet import nd as T
import numpy as np
from .common import _FLOATX, floatx, _EPSILON, image_dim_ordering, reset_uids, get_uid
from numbers import Number

_LEARNING_PHASE=1 
_EXECUTOR = None
_bind_values = {}

def learning_phase():
    # False = test, True = train
    return _LEARNING_PHASE

def set_learning_phase(value):
    global _LEARNING_PHASE
    if value not in {0, 1}:
        raise ValueError('Expected learning phase to be '
                         '0 or 1.')
    _LEARNING_PHASE = value

def _typename(t):
        if t == np.float16:
            return 'float16'
        elif t == np.float32:
            return 'float32'
        elif t == np.float64:
            return 'float64'
        elif t == np.uint8:
            return 'uint8'
        elif t == np.uint16:
            return 'uint16'
        elif t == np.int16:
            return 'int16'
        elif t == np.int32:
            return 'int32'
        elif t == np.int64:
            return 'int64'
        else:
            raise TypeError('unknown type')

class KerasTensor(object):
    def __init__(self, ndarray, name=None):
        if not isinstance(ndarray, mx.ndarray.NDArray):
            raise TypeError
        self.tensor = ndarray
        self._uses_learning_phase = False
        if name is not None:
            self.name_ = name
        else:
            self.name_ = _autogen_name('tensor')
        _bind_values[name] = ndarray
    
    @property
    def name(self):
        return self.name_

    @property
    def dtype(self):
        return _typename(self.tensor.dtype)

    @property
    def shape(self):
        return self.get_shape()

    def get_shape(self):
        return self.tensor.shape

    @property
    def symbol(self):
        return mx.sym.Variable(self.name, shape=self.shape, dtype=self.dtype)

    def __add__(self, other):
        return KerasTensor(mx.tensor.__add__(other))


class KerasSymbol(object):
    def __init__(self, symbol):
        if not isinstance(symbol, mx.symbol.Symbol):
            raise TypeError
        self.symbol = symbol
        self._uses_learning_phase = False

    @property
    def name(self):
        return self.symbol.name

    @property
    def dtype(self):
        return self.get_type()

    def get_shape(self):
        _, out_shape, _ = self.symbol.infer_shape_partial()
        return out_shape[0]

    def get_type(self):
        _, out_type, _ = self.symbol.infer_type()
        t = out_type[0]
        return _typename(t)

    def __add__(self, other):
        return KerasSymbol(
                mx.sym.broadcast_add(
                    lhs = self.symbol,
                    rhs = other.symbol))

    def __sub__(self, other):
        return KerasSymbol(
                mx.sym.broadcast_minus(
                    lhs = self.symbol,
                    rhs = other.symbol))

    def __div__(self, other):
        if isinstance(other, Number):
            return KerasSymbol(
                    self.symbol / other)
        else:
            return KerasSymbol(
                    self.symbol / other.symbol)

    def __mul__(self, other):
        if isinstance(other, Number):
            return KerasSymbol(
                    self.symbol * other)
        else:
            return KerasSymbol(
                    self.symbol * other.symbol)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __eq__(self, other):
        if isinstance(other, Number):
            return KerasSymbol(
                self.symbol.__eq__(other))
        else:
            return KerasSymbol(
                self.symbol.__eq__(other.symbol))

    def __str__(self):
        return "Symbol:" + self.symbol.name

def KerasVariable(name, shape, dtype):
    if dtype is None:
        dtype = floatx()
    v = mx.sym.Variable(name, shape=shape, dtype=dtype)
    ret = KerasSymbol(v)
    ret._uses_learning_phase = False
    ret._keras_shape = shape 
    return ret 

def clear_session():
    reset_uids()
    _EXECUTOR = None

def _autogen_name(prefix):
    return prefix + str(get_uid(prefix))

def dtype(x):
    '''Returns the dtype of a Keras tensor or variable, as a string.

    # Arguments
        x: Tensor or variable.

    # Returns
        String, dtype of `x`.

    '''
    return x.get_type()

def get_value(x):
    return eval(x)

def batch_get_value(xs):
    '''Returns the value of more than one tensor variable,
    as a list of Numpy arrays.
    '''
    return [get_value(x) for x in xs]

def get_variable_shape(x):
    return x.shape

def eval(x):
    if isinstance(x, KerasTensor):
        return x.tensor.asnumpy()
    elif isinstance(x, KerasSymbol):
        executor = x.symbol.simple_bind(mx.cpu())
        for v in executor.arg_dict:
            _bind_values[v].copyto(executor.arg_dict[v])
        outputs = executor.forward(is_train=_LEARNING_PHASE)
        return outputs[0].asnumpy()
    else:
        raise ValueError('value is not supported')

def variable(value, dtype=None, name=None):
    '''Instantiates a variable and returns it.

    # Arguments
        value: Numpy array, initial value of the tensor.
        dtype: Tensor type.
        name: Optional name string for the tensor.

    # Returns
        A variable instance (with Keras metadata included).
    '''
    if name is None:
        name = _autogen_name('variable')
    if dtype is None:
        dtype = floatx()
    if isinstance(value, float):
        value = np.array([value])
    ndarray = mx.nd.array(value, dtype=dtype)
    sym = KerasTensor(ndarray, name)
    return sym 

def count_params(x):
    '''Returns the number of scalars in a Keras variable.

    # Arguments
        x: Keras variable.

    # Returns
        Integer, the number of scalars in `x`.

    # Example
    ```python
        >>> kvar = K.zeros((2,3))
        >>> K.count_params(kvar)
        6
        >>> K.eval(kvar)
        array([[ 0.,  0.,  0.],
               [ 0.,  0.,  0.]], dtype=float32)
    ```
    '''
    shape = x.get_shape() 
    return np.prod([shape[i] for i in range(len(shape))])

def placeholder(shape=None, ndim=None, dtype=None, sparse=False, name=None):
    '''Instantiates a placeholder tensor and returns it.

    # Arguments
        shape: Shape of the placeholder
            (integer tuple, may include `None` entries).
        ndim: Number of axes of the tensor.
            At least one of {`shape`, `ndim`} must be specified.
            If both are specified, `shape` is used.
        dtype: Placeholder type.
        name: Optional name string for the placeholder.

    # Returns
        Tensor instance (with Keras metadata included).

    '''
    if name is None:
        name = _autogen_name('placeholder')
    if not shape:
        if ndim:
            shape = tuple([0 for _ in range(ndim)])
    else:
        shape = tuple([0 if x is None else x for x in shape])
    sym = KerasVariable(name, shape=shape, dtype=dtype)
    return sym

def not_equal(x, y):
    if isinstance(y, KerasSymbol):
        y = y.symbol
    return KerasSymbol(
            x.symbol.__ne__(y))

def equal(x, y):
    if isinstance(y, KerasSymbol):
        y = y.symbol
    return KerasSymbol(
            x.symbol.__eq__(y))

def shape(x):
    '''Returns the symbolic shape of a tensor or variable.

    # Arguments
        x: A tensor or variable.

    # Returns
        A symbolic shape (which is itself a tensor).
    '''
 #   if hasattr(x, '_keras_shape'):
 #       return tuple([0 if x is None else x for x in x._keras_shape])
    if isinstance(x, KerasSymbol):
        return x.get_shape()
    else:
        return None

def in_train_phase(x, alt):
    '''Selects `x` in train phase, and `alt` otherwise.
    Note that `alt` should have the *same shape* as `x`.
    '''
    if learning_phase() is 1:
        return x
    elif learning_phase() is 0:
        return alt
    # else: assume learning phase is a placeholder tensor.
    x = switch(learning_phase(), x, alt)
    x._uses_learning_phase = True
    return x


def in_test_phase(x, alt):
    '''Selects `x` in test phase, and `alt` otherwise.
    Note that `alt` should have the *same shape* as `x`.
    '''
    if learning_phase() is 1:
        return alt
    elif learning_phase() is 0:
        return x
    # else: assume learning phase is a placeholder tensor.
    x = switch(learning_phase(), alt, x)
    x._uses_learning_phase = True
    return x

def int_shape(x):
    '''Returns the shape of a Keras tensor or a Keras variable as a tuple of
    integers or None entries.

    # Arguments
        x: Tensor or variable.

    # Returns
        A tuple of integers (or None entries).

    '''
    s = shape(x)
    if s is None:
        return None
    else:
        return tuple([i.__int__() for i in s])

def ndim(x):
    '''Returns the number of axes in a tensor, as an integer.

    # Arguments
        x: Tensor or variable.

    # Returns
        Integer (scalar), number of axes.

    '''
    s = shape(x)
    if s is None:
        return None
    else:
        return len(s)

def cast(x, dtype):
    '''Casts a tensor to a different dtype and returns it.

    You can cast a Keras variable but it still returns a Keras tensor.

    # Arguments
        x: Keras tensor (or variable).
        dtype: String, either (`'float16'`, `'float32'`, or `'float64'`).

    # Returns
        Keras tensor with dtype `dtype`.
    '''
    return KerasSymbol(
            mx.sym.Cast(data=x.symbol, dtype=dtype))

def random_uniform(shape, low=0.0, high=1.0, dtype=None, seed=None):
    return KerasVariable(
            random_uniform_variable(shape, low, high, dtype=dtype, seed=seed).symbol)

def random_uniform_variable(shape, low, high, dtype=None,
                            name=None, seed=None):
    '''Instantiates an Keras variable filled with
    samples drawn from a uniform distribution and returns it.

    # Arguments
        shape: Tuple of integers, shape of returned Keras variable.
        low: Float, lower boundary of the output inteval.
        high: Float, upper boundary of the output interval.
        dtype: String, dtype of returned Keras variable.
        name: String, name of returned Keras variable.
        seed: Integer, random seed.

    # Returns
        A Keras variable, filled with drawn samples.
    '''
    #if seed:
    #    mx.random.seed(seed)
    if dtype is None:
        dtype = floatx()
    value = mx.random.uniform(low, high, shape)
    if name is None:
        name = _autogen_name('randinit') 
    return KerasTensor(value, name)

def zeros(shape, dtype=None, name=None):
    '''Instantiates an all-zeros variable and returns it.

    # Arguments
        shape: Tuple of integers, shape of returned Keras variable
        dtype: String, data type of returned Keras variable
        name: String, name of returned Keras variable

    # Returns
        A variable (including Keras metadata), filled with `0.0`.

    '''
    if dtype is None:
        dtype = floatx()
    value = mx.nd.zeros(shape, dtype=dtype)
    if name is None:
        name = _autogen_name('zeroinit') 
    return KerasTensor(value, name)

# LINEAR ALGEBRA

def dot(x, y):
    '''Multiplies 2 tensors (and/or variables) and returns a *tensor*.
    When attempting to multiply a ND tensor
    with a ND tensor, it reproduces the Theano behavior.
    (e.g. (2, 3).(4, 3, 5) = (2, 4, 5))

    # Arguments
        x: Tensor or variable.
        y: Tensor or variable.

    # Returns
        A tensor, dot product of `x` and `y`.
    '''
    print x.symbol
    print y.symbol
    return KerasSymbol(mx.sym.dot(lhs=x.symbol, rhs=y.symbol))

def relu(x, alpha=0., max_value=None):
    '''Rectified linear unit

    # Arguments
        alpha: slope of negative section.
        max_value: saturation threshold.
    '''
    if alpha != 0.:
        ret =  mx.sym.LeakyReLU(data=x.symbol,
            slope = alpha)
    else:
        ret = mx.sym.Activation(data=x.symbol,
            act_type='relu')
    return KerasSymbol(ret)

def is_sparse(tensor):
    return False

def _normalize_axis(axis, ndim):
    if isinstance(axis, tuple):
        axis = list(axis)
    if isinstance(axis, list):
        for i, a in enumerate(axis):
            if a is not None and a < 0:
                axis[i] = a % ndim
    else:
        if axis is not None and axis < 0:
            axis = axis % ndim
    return axis

def argmax(x, axis=-1):
    axis = _normalize_axis(axis, ndim(x)) 
    if axis != None:
        ret = mx.sym.argmax(data=x.symbol, axis=axis)
    else:
        ret = mx.sym.argmax(data=x.symbol)
    return KerasSymbol(ret)

def mean(x, axis=None, keepdims=False):
    axis = _normalize_axis(axis, ndim(x)) 
    if axis != None:
        ret = mx.sym.sum(data=x.symbol, axis=axis, keepdims=keepdims)
    else:
        ret = mx.sym.sum(data=x.symbol, keepdims=keepdims)
    ret = ret / count_params(x)
    return KerasSymbol(ret)

def square(x):
    '''Element-wise square.
    '''
    ret = mx.sym.square(data=x.symbol) 
    return KerasSymbol(ret)


def dropout(x, level, noise_shape=None, seed=None):
    '''Sets entries in `x` to zero at random,
    while scaling the entire tensor.

    # Arguments
        x: tensor
        level: fraction of the entries in the tensor
            that will be set to 0.
        noise_shape: shape for randomly generated keep/drop flags,
            must be broadcastable to the shape of `x`
        seed: random seed to ensure determinism.
    '''
    return KerasSymbol(
            mx.sym.Dropout(data=x.symbol, p=level))

def transpose(x):
    '''Transposes a tensor and returns it.

    # Arguments
        x: Tensor or variable.

    # Returns
        A tensor.
    '''
    return KerasSymbol(
            mx.sym.transpose(data=x.symbol))

def gradients(loss, variables):
    '''Returns the gradients of `variables` (list of tensor variables)
    with regard to `loss`.
    '''
    #TODO
    return loss

def softmax(x):
    return KerasSymbol(
            mx.sym.SoftmaxOutput(data=x.symbol))

def categorical_crossentropy(output, target, from_logits=False):
    if not from_logits:
        return output
    return 


class Function(object):
    def __init__(self, inputs, output, updates=[], **kwargs):
        print 'inputs', [x.name for x in inputs]
        print 'output', [x.name for x in output]
        print 'updates', updates

    def __call__(self, inputs):
        print 'call', inputs
        return None

def function(inputs, outputs, updates=[], **kwargs):
    return Function(inputs, outputs, updates=updates, **kwargs)
