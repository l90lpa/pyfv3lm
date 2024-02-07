import jax.numpy as jnp

class HashableArrayWrapper:
  def __init__(self, val):
    self.val = val
  def __hash__(self):
    return int(jnp.sum(self.val))
  def __eq__(self, other):
    return (isinstance(other, HashableArrayWrapper) and
            jnp.all(jnp.equal(self.val, other.val)))

def pack_hashable_array(obj):
    if isinstance(obj, HashableArrayWrapper):
        return obj
    return HashableArrayWrapper(obj)

def unpack_hashable_array(obj):
    if isinstance(obj, HashableArrayWrapper):
        return obj.val
    return obj
