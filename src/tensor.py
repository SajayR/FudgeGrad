import numpy as np
"""
TODO: 
-Broadcast handling

"""
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
    def __pow__(self, power):
        parents=(self, power)
        return(Tensor(value=self.data.__pow__(power), parents=parents, gen_op="pow", requires_grad=self.requires_grad))
        
    def _matmul_(self, multiplicand: "Tensor"):
        assert self.shape[1]==multiplicand.shape[0], f"My guy take some linear alg classes (axb)X(bxc). Inner dimensions of both dimensions dont match {self.shape}, {multiplicand.shape}"
        parents=(self, multiplicand)
        return Tensor(value=self.data@multiplicand.data, parents=parents, gen_op="matmul", requires_grad=(self.requires_grad or multiplicand.requires_grad))
        
    """Grad Operations"""
    def _add_backward(self, alt_parents: tuple, parent: "Tensor", forward_grad: "Tensor"):
        return forward_grad
    def _mul_backward(self, alt_parents: tuple, parent: "Tensor", forward_grad: "Tensor"):
        if isinstance(alt_parents[0], Tensor): return alt_parents[0] * forward_grad
        else: return forward_grad*float(alt_parents[0])

    def _matmul_backward(self, alt_parents: tuple, parent: "Tensor", forward_grad: "Tensor"):
        pass

    def _pow_backward(self, alt_parents: tuple, parent: "Tensor", forward_grad: "Tensor"):
        power=float(alt_parents[0])
        return Tensor(value=power*(parent.data**(power-1)*forward_grad.data))



        
    def _grad(self, forward_grad: "Tensor"=None): 
        assert self.requires_grad, 'Grad is not enabled for this tensor'
        backward_op = f"_{self.gen_op.lower()}_backward"
        if forward_grad is None:
            for i, parent in enumerate([p for p in self.parents if isinstance(p, Tensor)]) :   
                self.parents[i].grad += getattr(self, backward_op)(alt_parents=(tuple(self.parents[:i]+self.parents[i+1:])), parent=self.parents[i], forward_grad=Tensor(value=np.ones(self.shape)))
                if self.parents[i].gen_op is not None:
                    self.parents[i]._grad(self.parents[i].grad)
        else:
            for i, parent in enumerate([p for p in self.parents if isinstance(p, Tensor)]):   
                self.parents[i].grad += getattr(self, backward_op)(alt_parents=(tuple(self.parents[:i]+self.parents[i+1:])), parent=self.parents[i], forward_grad=forward_grad)
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

    a = Tensor(value=np.array([2.0]), requires_grad=True)
    b = a ** 2    # b = 4
    c = b * 3     # c = 12

    print(c)
    c._grad()
    print(c.grad, b.grad, a.grad)

    # Let's minimize (x^2 + 2)^2 or something
    x = Tensor(value=np.array([5.0]), requires_grad=True)  # start at x=5
    learning_rate = 0.1

    for step in range(100):
        # Forward pass
        y = (x + 2) ** 2
        
        # Backward pass
          # Important! Clear old grads
        x._zero_grad()
        y._grad()
        
        # Gradient descent step
        x.data = x.data - learning_rate * x.grad.data
        
        if step % 10 == 0:
            print(f"Step {step}, x = {x}, loss = {y}")

    # Print gradients - let's see if they match what we calculated by hand!