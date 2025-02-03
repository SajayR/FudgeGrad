import numpy as np
class Tensor():
    def __init__(self, shape: tuple=None, dtype=float, value: np.ndarray=None, requires_grad: bool=False, parents: tuple=None, gen_op: str=None) -> None:
        assert shape is not None or value is not None, "Either shape or data has to be passed for tensor creation"
        assert ((shape is not None) and (value is None)) or ((shape is None) and (value is not None))
        if value is not None:
            assert isinstance(value, np.ndarray)
            self.data = value.copy()
        else: 
            self.data = np.empty(shape=shape, dtype=dtype)      
        self.dtype = self.data.dtype
        self.shape = self.data.shape
        self.size = self.data.size
        self.requires_grad = requires_grad
        if parents is not None: self.parents = parents
        if gen_op is not None: self.gen_op = gen_op
    def __repr__(self) -> str: return(f"Tensor Value: {self.data}")
    def __add__(self, additive: "Tensor | float"):
        if isinstance(additive, Tensor):
            parents=(self, additive)
            return Tensor(value=np.add(self.data, additive.data), parents=parents, gen_op="Add", requires_grad=(self.requires_grad or additive.requires_grad))
        else:
            parents=(self)
            return Tensor(value=np.add(self.data, additive), parents=parents, gen_op="Add", requires_grad=self.requires_grad)

    






if __name__=="__main__":
    array = np.ones((2,3))
    a = Tensor(shape=(2,3))
    b = Tensor(value=array)
    print(a)
    print(a+b)

        