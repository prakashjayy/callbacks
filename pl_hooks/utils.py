__all__ = ['import_module', 'locate_cls']

import pydoc
from loguru import logger
from functools import partial

def import_module(d, parent=None, **default_kwargs):
    # copied from
    kwargs = d.copy()
    object_type = kwargs.pop("type")
    for name, value in default_kwargs.items():
        kwargs.setdefault(name, value)

    try:
        if parent is not None:
            module = getattr(parent, object_type)(**kwargs)  # skipcq PTC-W0034
        else:
            module = pydoc.locate(object_type)(**kwargs)
    except Exception as e:
        logger.error(f"Cannot load {name}. Error: {str(e)}")
    return module

def locate_cls(transforms: dict, return_partial=False):
    name = transforms["__class_fullname__"]
    targs = {k: v for k, v in transforms.items() if k != "__class_fullname__"}
    try:
        if return_partial:
            transforms = partial(pydoc.locate(name), **targs)
        else:
            transforms = pydoc.locate(name)(**targs)
    except Exception as e:
        logger.error(f"Cannot load {name}. Error: {str(e)}")
    return transforms
