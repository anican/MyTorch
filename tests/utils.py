from collections import defaultdict
import numpy as np
import torch

from nn.modules import Linear

TOLERANCE = 1e-6


def to_numpy(array):
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    elif isinstance(array, dict):
        return {key: to_numpy(val) for key, val in array.items()}
    else:
        return np.asarray(array)


numpy_dtype_to_pytorch_dtype_warn = False


def numpy_dtype_to_pytorch_dtype(numpy_dtype):
    global numpy_dtype_to_pytorch_dtype_warn
    # Extremely gross conversion but the only one I've found
    numpy_dtype = np.dtype(numpy_dtype)
    if numpy_dtype == np.uint32:
        if not numpy_dtype_to_pytorch_dtype_warn:
            print("numpy -> torch dtype uint32 not supported, using int32")
            numpy_dtype_to_pytorch_dtype_warn = True
        numpy_dtype = np.int32
    return torch.from_numpy(np.zeros(0, dtype=numpy_dtype)).detach().dtype


from_numpy_warn = defaultdict(lambda: False)


def from_numpy(np_array):
    global from_numpy_warn
    if isinstance(np_array, list):
        try:
            np_array = np.stack(np_array, 0)
        except ValueError:
            np_array = np.stack([from_numpy(val) for val in np_array], 0)
    elif isinstance(np_array, dict):
        return {key: from_numpy(val) for key, val in np_array.items()}
    np_array = np.asarray(np_array)
    if np_array.dtype == np.uint32:
        if not from_numpy_warn[np.uint32]:
            print("numpy -> torch dtype uint32 not supported, using int32")
            from_numpy_warn[np.uint32] = True
        np_array = np_array.astype(np.int32)
    elif np_array.dtype == np.dtype("O"):
        if not from_numpy_warn[np.dtype("O")]:
            print("numpy -> torch dtype Object not supported, returning numpy array")
            from_numpy_warn[np.dtype("O")] = True
        return np_array
    elif np_array.dtype.type == np.str_:
        if not from_numpy_warn[np.str_]:
            print("numpy -> torch dtype numpy.str_ not supported, returning numpy array")
            from_numpy_warn[np.str_] = True
        return np_array
    return torch.from_numpy(np_array)


def assign_linear_weights(mytorch_layer: Linear, torch_layer: torch.nn.Linear):
    """
    Initialize a pytorch linear layer with the same parameters as that of a mytorch linear layer.

    :param mytorch_layer: Instance of a mytorch linear layer whose data is to be replicated.
    :param torch_layer: Instance of a pytorch linear layer whose parameters are to be set with those of `mytorch_layer`
    :return:
    """
    with torch.no_grad():
        torch_layer.weight[:] = torch.from_numpy(mytorch_layer.weight.data).transpose(0, 1)
        if mytorch_layer.is_bias:
            torch_layer.bias[:] = torch.from_numpy(mytorch_layer.bias.data)


def assert_close(val1, val2, atol=TOLERANCE, rtol=1e-5):
    # print(val1)
    # print(val2)
    val1 = to_numpy(val1)
    val2 = to_numpy(val2)
    val1[np.abs(val1) < 1e-7] = 0
    val2[np.abs(val2) < 1e-7] = 0
    if not np.allclose(val1, val2, atol=atol):
        lhs = np.abs(val1 - val2)
        rhs = atol + rtol * np.abs(val2)
        diff = rhs - lhs
        loc = np.unravel_index(np.argmin(diff), diff.shape)
        min_diff = np.min(diff)
        mask = diff == min_diff
        num_errors = np.sum(diff < 0)
        assert False, ('Largest diff %s vs. %s array shape %s at %s, num total failures %s, atol %f rtol %f' %
                        (val1[loc], val2[loc], mask.shape, loc, num_errors, atol, rtol))


def check_linear_match(mytorch_layer: Linear, torch_layer: torch.nn.Linear, tolerance=TOLERANCE):
    my_weight = mytorch_layer.weight.data
    torch_weight = to_numpy(torch_layer.weight.transpose(0, 1))
    assert_close(my_weight, torch_weight, tolerance)

    if mytorch_layer.is_bias:
        my_bias = mytorch_layer.bias.data
        torch_bias = to_numpy(torch_layer.bias)
        assert_close(my_bias, torch_bias, tolerance)


def check_linear_grad_match(mytorch_layer: Linear, torch_layer: torch.nn.Linear, tolerance=TOLERANCE):
    my_weight_grad = mytorch_layer.weight.grad
    torch_weight_grad = torch_layer.weight.grad
    assert_close(my_weight_grad, torch_weight_grad, TOLERANCE)

    if mytorch_layer.is_bias:
        my_bias_grad = mytorch_layer.bias.grad
        torch_bias_grad = torch_layer.bias.grad
        assert_close(my_bias_grad, torch_bias_grad, tolerance)
