# analogous to layer.py in the deep learning homework
# This class was written by Daniel Gordon/Aaron Walsman for the deep learning class, I have adjusted it a little for
# personal use in MyTorch
from abc import ABC
from collections import OrderedDict, deque
import numpy as np
from nn import Parameter
from nn.modules import _DummyLayer
from typing import Callable, List, Tuple, Iterable, Dict, Optional, Union


class Module(object):
    """
    Base class for all layer modules.
    """
    def __init__(self, parent: Optional[Union["Module", Tuple["Module", ...]]] = None):
        assert (
                parent is None or isinstance(parent, Module) or isinstance(parent, Tuple)
        ), "Parents must be a Layer, a list/tuple of Layers, or None"

        self._parent = self._fix_parent(parent)

    @property
    def name(self) -> str:
        return type(self).__name__

    @staticmethod
    def _fix_parent(parent):
        if parent is not None:
            if isinstance(parent, Module):
                if hasattr(parent, "final_layer"):
                    parent = parent.final_layer
            else:
                final_layers = []
                for pp in parent:
                    if hasattr(pp, "final_layer"):
                        pp = pp.final_layer
                    final_layers.append(pp)
                parent = tuple(final_layers)
        return parent

    @property
    def parent(self) -> Optional[Union[Tuple["Module", ...], "Module"]]:
        self._parent = self._fix_parent(self._parent)
        return self._parent

    @parent.setter
    def parent(self, val: Union[Tuple["Module", ...], "Module"]):
        self.set_parent(val)

    def set_parent(self, val):
        self._parent = val

    @property
    def parents(self) -> Optional[Tuple["Module", ...]]:
        if self.parent is None:
            return None
        if isinstance(self.parent, Tuple):
            return self.parent
        else:
            return (self.parent,)

    def forward(self, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError

    def backward(self, *args, **kwargs) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        raise NotImplementedError

    def vars(self):
        for obj, val in vars(self).items():
            if obj == "_parent":
                continue
            yield (obj, val)

    def state_dict(self, prefix="") -> Dict[str, Parameter]:
        state_dict = OrderedDict()
        for obj, val in self.vars():
            if isinstance(val, Module):
                state_dict.update(val.state_dict(obj + "."))
            elif isinstance(val, Parameter):
                state_dict[prefix + obj] = val
        return state_dict

    def parameters(self) -> Iterable[Parameter]:
        return self.state_dict().values()

    def children(self) -> Iterable["Module"]:
        for obj, val in self.vars():
            if isinstance(val, Module):
                yield val

    def own_parameters(self) -> Iterable[Parameter]:
        params = []
        for obj, val in self.vars():
            if isinstance(val, Parameter):
                params.append(val)
        return params

    def share_parameters(self, other: "Module"):
        assert type(self) == type(other)
        self_vars = sorted(self.vars())
        other_vars = sorted(other.vars())
        for ((self_obj, self_val), (other_obj, other_val)) in zip(self_vars, other_vars):
            if isinstance(self_val, Module):
                self_val.share_parameters(other_val)
            elif isinstance(self_val, Parameter):
                assert self_val.data.shape == other_val.data.shape
                setattr(self, self_obj, other_val)

    def initialize(self, initializer: Optional[Callable[[Parameter], None]] = None):
        if initializer is not None:
            for param in self.own_parameters():
                initializer(param)
        for val in self.children():
            val.initialize()

    def selfstr(self) -> str:
        """
        Overload this function to print the specifics for this layer.
        See LinearLayer for an example.
        """
        return ""

    def _total_str(self, depth=0) -> List[str]:
        from nn import _DummyLayer
        total_str_arr = []
        for obj, val in self.vars():
            if isinstance(val, Module):
                if isinstance(val, _DummyLayer):
                    total_str_arr.extend(val._total_str(depth))
                else:
                    total_str_arr.append(("(" + obj + "): " + val.name + ": " + val.selfstr(), depth))
                    # total_str_arr.append((val.selfstr(), depth + 1))
                    total_str_arr.extend(val._total_str(depth + 1))
        return total_str_arr

    def __str__(self):
        total_str_arr = list()
        total_str_arr.append((self.name + ": " + self.selfstr(), 0))
        # total_str_arr.append((self.selfstr(), 1))
        total_str_arr.extend(self._total_str(1))
        filtered_strs = []
        for val in total_str_arr:
            if val is None or len(val) == 0:
                continue
            string, depth = val
            if string is None or len(string) == 0:
                continue
            filtered_strs.append((string, depth))
        strs = [" " * (depth * 4) + string for string, depth in filtered_strs]
        return "\n".join(strs)

    __repr__ = __str__

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class MetaModule(Module, ABC):
    def __init__(self, parent=None):
        if parent is None:
            parent = _DummyLayer()
        super(MetaModule, self).__init__(parent)

    @property
    def final_layer(self):
        raise NotImplementedError

    def set_parent(self, val):
        if isinstance(self._parent, _DummyLayer):
            self._parent.set_parent(val)
        else:
            self._parent = val

    def backward(self, previous_partial_gradients=None) -> np.ndarray:
        if previous_partial_gradients is not None:
            gradient = self.final_layer.backward(previous_partial_gradients)
        else:
            gradient = self.final_layer.backward()

        # Build Computation Graph
        frontier = deque([self.final_layer])
        graph = {}
        while frontier:
            node = frontier.popleft()
            if node.parents is None:
                continue
            for parent in node.parents:
                if parent not in graph:
                    graph[parent] = set()
                graph[parent].add(node)
                frontier.append(parent)

        # Topological Sort
        order = []
        frontier = deque([self.final_layer])
        while frontier:
            node = frontier.popleft()
            order.append(node)
            if node.parents is None:
                continue
            for parent in node.parents:
                graph[parent].remove(node)
                if len(graph[parent]) == 0:
                    frontier.append(parent)

        grad_dict = {}
        self._assign_parent_grads(self.final_layer, gradient, grad_dict)
        # Ignore loss layer because already computed
        order = order[1:]
        # Send gradients backwards
        for layer in order:
            output_grad = layer.backward(grad_dict[layer])
            if layer.parents is not None:
                self._assign_parent_grads(layer, output_grad, grad_dict)
        return output_grad

    @staticmethod
    def _assign_parent_grads(layer, grad, grad_dict):
        assert isinstance(grad, np.ndarray) or isinstance(grad, tuple), (
                "grad should be a nparray or a tuple of nparrays but was %s." % type(grad).__name__)
        assert isinstance(layer.parent, tuple) == isinstance(grad, tuple), (
            "Gradients should be a tuple iff there are multiple parents.")
        if not isinstance(grad, tuple):
            grad = (grad,)
        for parent, grad in zip(layer.parents, grad):
            if parent in grad_dict:
                grad_dict[parent] = grad_dict[parent] + grad
            else:
                grad_dict[parent] = grad
