import json
import numpy as np

from dataclasses import is_dataclass, fields


def ndarray_encoder(obj):
    assert isinstance(obj, np.ndarray) or obj is None

    if obj is not None: 
        assert obj.flags['C'] or obj.flags['F']

        order = 'C' if obj.flags['C'] else 'F'
        return dict(__ndarray__=obj.tolist(),
                    dtype=str(obj.dtype),
                    shape=obj.shape,
                    order=order)
    else:
        return None


def ndarray_decoder(dct):
    assert (isinstance(dct, dict) and '__ndarray__' in dct) or dct is None
    return np.array(dct['__ndarray__'], dtype=dct['dtype'], order=dct['order']) if dct is not None else None


def to_dct(obj, encoders={}):
    """
    Used for encoding a dataclass into a dictionary.

    Example:

    ```
    y = to_dct(x, encoders={np.ndarray: ndarray_encoder})
    ```
    """
    assert is_dataclass(obj)
    return _to_dct(obj, encoders)

def _to_dct(obj, encoders={}):
    if is_dataclass(obj):
        dct = {}
        for field in fields(obj):
            field_obj = getattr(obj, field.name)
            print(f"name={field.name}, type={field.type}")
            if encoder := encoders.get(field.type):
                dct[field.name] = encoder(field_obj)
            elif is_dataclass(field.type):
                dct[field.name] = _to_dct(field_obj, encoders)
            else:
                dct[field.name] = field_obj
        return dct
    return None

def from_dct(dct, mold, decoders={}):
    """
    Used for decoding a dictionary into a dataclass.

    Example:

    ```
    dx = from_dct(y, x, decoders={np.ndarray: ndarray_decoder})
    ```
    """
    assert isinstance(dct, dict) and is_dataclass(mold)
    return _from_dct(dct, mold, decoders)

def _from_dct(dct, mold, decoders={}):
    if is_dataclass(mold):
        kwargs = {}
        for field in fields(mold):
            field_dct = dct[field.name]
            if decoder := decoders.get(field.type):
                kwargs[field.name] = decoder(field_dct)
            elif is_dataclass(field.type):
                dct[field.name] = _from_dct(field_dct, field, decoders)
            else:
                dct[field.name] = field_dct
        return mold(**kwargs) if isinstance(mold, type) else type(mold)(**kwargs)
    return None