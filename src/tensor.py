import numpy as np

class Tensor():
    def __init__(self, shape: tuple=None, dtype=float, value: np.ndarray=None, requires_grad: bool=False, parents: tuple=None, gen_op: str=None) -> None:
        assert shape is not None or value is not None, "Either shape or data has to be passed for tensor creation"
        assert ((shape is not None) and (value is None)) or ((shape is None) and (value is not None)), "Only one of value or shape can be defined. If a numpy array is provided as the value, the shape will be replicated."
        if value is not None:
            assert isinstance(value, np.ndarray)
            self.data = value.copy()
        else: self.data = np.empty(shape=shape, dtype=dtype)      
        self.dtype = self.data.dtype
        self.shape = self.data.shape
        self.size = self.data.size
        self.requires_grad = requires_grad
        self.gen_op=None
        self.parents=None
        if parents is not None: self.parents = parents
        if gen_op is not None: self.gen_op = gen_op
        if self.requires_grad:
            self.grad = Tensor(value=np.zeros(self.shape))
        else: self.grad = None

    def __repr__(self) -> str: return(f"{self.data}")

    """Operations"""
    def __add__(self, additive: "Tensor | float"): #float carries no grad
        if isinstance(additive, Tensor):
            parents=(self, additive)
            return Tensor(value=np.add(self.data, additive.data), parents=parents, gen_op="add", requires_grad=(self.requires_grad or additive.requires_grad))
        else:
            parents=(self,)
            return Tensor(value=np.add(self.data, additive), parents=parents, gen_op="add", requires_grad=self.requires_grad)
        
    def __mul__(self, multiplicand: "Tensor | float"): #doing elementwise here, not cross fuck cross
        if isinstance(multiplicand, Tensor):
            parents=(self, multiplicand)
            return Tensor(value=self.data*multiplicand.data, parents=parents, gen_op="mul", requires_grad=(self.requires_grad or multiplicand.requires_grad))
        else:
            parents=(self, multiplicand)
            return Tensor(value=self.data*multiplicand, parents=parents, gen_op="mul", requires_grad=self.requires_grad)
        
    """Grad Operations"""
    def _add_backward(self, alt_parents: tuple, parent: "Tensor", forward_grad: "Tensor"):
        return forward_grad
    
    def _mul_backward(self, alt_parents: tuple, parent: "Tensor", forward_grad: "Tensor"):
        return alt_parents[0] * forward_grad
        
    def _grad(self, forward_grad: "Tensor"=None): 
        assert self.requires_grad, 'Grad is not enabled for this tensor'
        backward_op = f"_{self.gen_op.lower()}_backward"
        print(f"Backward op {backward_op}")
        if forward_grad is None:
            for i, parent in enumerate(self.parents):   
                self.parents[i].grad += getattr(self, backward_op)(alt_parents=(self.parents[:i]+self.parents[i+1:]), parent=self.parents[i], forward_grad=Tensor(value=np.ones(self.shape)))
                if self.parents[i].gen_op is not None:
                    self.parents[i]._grad(self.parents[i].grad)
        else:
            for i, parent in enumerate(self.parents):   
                self.parents[i].grad += getattr(self, backward_op)(alt_parents=(self.parents[:i]+self.parents[i+1:]), parent=self.parents[i], forward_grad=forward_grad)
                if self.parents[i].gen_op is not None:
                    self.parents[i]._grad(self.parents[i].grad)

    def _zero_grad(self,):
        assert self.requires_grad, "Grad is not enabled for this tensor"
        self.grad = Tensor(value=np.zeros(self.shape))


    






if __name__ == "__main__":
    # Create tensors
    a = Tensor(value=np.array([2.0]), requires_grad=True)
    b = Tensor(value=np.array([3.0]), requires_grad=True)
    
    c = a * b    # c = 6
    d = c + 1    # d = 7
    e = d * 2    # e = 14   # should be 6
    

    # Now backprop and check gradients
    print(e)
    e._grad()
    print(c.grad)
    print(a.grad)
    print(b.grad)
    #print(L._grad())    # this should populate gradients


    # Print gradients - let's see if they match what we calculated by hand!