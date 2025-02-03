import numpy as np
from grad_ops import grad_operations
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
        if parents is not None: self.parents = parents
        if gen_op is not None: self.gen_op = gen_op
        if self.requires_grad:
            self.grad = Tensor(value=np.zeros(self.shape))
        else: self.grad = None

    def __repr__(self) -> str: return(f"{self.data}")
    def __add__(self, additive: "Tensor | float"): #float carries no grad
        if isinstance(additive, Tensor):
            parents=(self, additive)
            return Tensor(value=np.add(self.data, additive.data), parents=parents, gen_op="add", requires_grad=(self.requires_grad or additive.requires_grad))
        else:
            parents=(self,)
            return Tensor(value=np.add(self.data, additive), parents=parents, gen_op="add", requires_grad=self.requires_grad)
        
    def __mul__(self, multiplicand: "Tensor | float"): #doing dot product here, not cross
        if isinstance(multiplicand, Tensor):
            parents=(self, multiplicand)
            return Tensor(value=self.data*multiplicand.data, parents=parents, gen_op="mul", requires_grad=(self.requires_grad or multiplicand.requires_grad))
        else:
            parents=(self,)
            return Tensor(value=self.data*multiplicand, parents=parents, gen_op="mul", requires_grad=self.requires_grad)
        
    def _grad(self, forward_grad: Tensor=None): 
        assert self.requires_grad, 'Grad is not enabled for this tensor'
        self.local_grad_op = grad_operations[self.gen_op] #what operation generated this shit's gradient formula 
        #local_grad_op is a func that returns the gradient vector
        #local_grad_op takes in the rest of the parents, and the forward_grad
        if forward_grad is None:
            for i, parent in enumerate(self.parents):   
                self.parents[i].grad += self.local_grad_op((self.parents[:i]+self.parents[i+1:]), forward_grad=Tensor(value=np.ones(self.shape)))
        else:
            for i, parent in enumerate(self.parents):   
                self.parents[i].grad += self.local_grad_op((self.parents[:i]+self.parents[i+1:]), forward_grad=forward_grad)

                
    







    def _zero_grad(self,):
        assert self.requires_grad, "Grad is not enabled for this tensor"
        self.grad = Tensor(value=np.zeros(self.shape))


    






if __name__=="__main__":
    array = np.ones((2,3))
    a = Tensor(shape=(2,3))
    b = Tensor(value=array)
    print(a)
    print(a+b)

        