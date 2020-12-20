"""Convenience functions for working with DataJoint"""

import re
from importlib import import_module
from inspect import isclass
from typing import Any, ForwardRef, _GenericAlias
import datajoint as dj
from .dbtables import QuantityAdapter
dj.errors._switch_adapted_types(True)

QUANTITY_ADAPTER = None

def schematize(cls, schema: dj.schema):
    """Take a Python class and build a Datajoint table from it.

    Params:
        cls: The class to convert into a DataJoint table
        schema: The schema in which to add the table

    Returns:
        cls: The schematized class (now for use with DataJoint)
    """
    cls = type(cls.__name__, (dj.Manual, cls, object), {})
    cls = set_dj_definition(cls)
    global QUANTITY_ADAPTER
    if QUANTITY_ADAPTER:
        schema.context.update({'QUANTITY_ADAPTER': QUANTITY_ADAPTER})
    cls = schema(cls)
    return cls

def create_quantity_adapter() -> None:
    """ Create an datajoint adapter class, `QuantityAdapter`, that puts and gets 
        Python Quantity objects to and from the datajoint database server.
        The adapter will be assigned to the global variable `QUANTITY_ADAPTER`
        in this module.
    """

    global QUANTITY_ADAPTER
    QUANTITY_ADAPTER = QuantityAdapter()


def handle_dict(cls, _type_map: dict, attr: Any, type_hint: _GenericAlias):
    """Using master-part relation to store a dict. It is assumed that 
        the type of keys have corresponding tables in the database.

        It is assumed that values of the dict are:
        primitive type which is in `_type_map`
        OR
        `quantities.Quantity` instance.

    Args:
        _type_map (dict): A map that maps type hint to data type that accepted by datajoint.
        attr (Any): Variable name of the dict.
        type_hint (typing._GenericAlias): Required to be a type hint like `Dict[TypeA, int]`.
                                            A type hint of `dict` will cause an exception.

    Returns:
        type: `cls` that contains a part class for the keys of the dict.
    """

    # For example, components: Dict[ClassA, int] = {a: 1, b: 2}
    # key_cls_name will be "ClassA"
    # part_cls_name will be "Component", 
    # note that the "s" at the end of the dict name will be removed.

    part_cls_name = attr[0].upper() + attr[1:]
    part_cls_name = part_cls_name[:-1] if part_cls_name[-1] == 's' else part_cls_name

    key_type = type_hint.__args__[0]
    value_type = type_hint.__args__[1]

    key_cls_name = key_type.__forward_arg__ if isinstance(key_type, ForwardRef) else key_type.__name__
    value_type = value_type.__forward_arg__ if isinstance(value_type, ForwardRef) else value_type.__name__

    if value_type == 'Quantity':
        if not QUANTITY_ADAPTER:
            create_quantity_adapter()
        value_type = '<QUANTITY_ADAPTER>'
    else:
        assert value_type in _type_map
        value_type = _type_map[value_type]

    part_cls = type(
        part_cls_name,
        (dj.Part, object),
        {
            "definition": "\n-> %s\n-> %s\n---\nvalue = NULL : %s"
            % (cls.__name__, key_cls_name, value_type)
        }
    )
    cls_dict = dict(vars(cls))
    cls_dict[part_cls_name] = part_cls
    cls = type(cls.__name__, tuple(cls.__bases__), {part_cls_name: part_cls})
    return cls

def set_dj_definition(cls, type_map: dict = None) -> None:
    """Set the definition property of a class by inspecting its attributes.

    Params:
        cls: The class whose definition attribute should be set
        type_map: Optional additional type mappings
    """
    # A mapping between python types and DataJoint types
    _type_map = {
        "int": "int", 
        "str": "varchar(256)", 
        "float": "float",
        "Quantity": "float",
        "datetime": "datetime", 
        "datetime.datetime": "datetime", 
        "bool": "tinyint",
        "list": "longblob",
    }
    # A list of python types which have no DataJoint
    # equivalent and so are unsupported
    unsupported = [dict]
    if type_map:
        _type_map.update(type_map)
    dj_def = "%s_id: int auto_increment\n---\n" % cls.__name__.lower()
    cls_items = cls.__annotations__.items()
    for attr, type_hint in cls_items:
        if type_hint in unsupported:
            continue
        name = getattr(type_hint, "__name__", type_hint)
        default = getattr(cls, attr)            
        
        if isinstance(default, str):
            default = '"%s"' % default
        elif isinstance(default, bool):
            default = int(default)
        else:
            default = "NULL"

        if getattr(type_hint, '_name', "") == 'Dict':
            cls = handle_dict(cls, _type_map, attr, type_hint)
            continue
        elif name in _type_map:
            dj_def += "%s = %s : %s\n" % (attr, default, _type_map[name])
        else:
            dj_def += "-> %s\n" % name
    cls.definition = dj_def
    return cls


def import_classes(module_name: str, match: str = None) -> dict:
    """Import all classes from the named module'

    Params:
        module_name (str): Name of the module (e.g. 'pyrfume.odorants')
        match (str): Optional regex string for class names to match.

    Returns:
        dict: A dictionary of full class names and the classes themselves"""
    classes = {}
    # Import the module and iterate through its attributes
    module = import_module(module_name)
    for attr_name in dir(module):
        if match and not re.search(match, attr_name):
            continue
        attr = getattr(module, attr_name)
        if isclass(attr):
            # Add the class to this package's variables
            classes[attr_name] = attr
    return classes


# Demonstration of setting the definition.
# A demonstration of adding to the schema requires an active schema.
if __name__ == "__main__":

    class Stuff:
        """A regular Python class that does not use DataJoint.
        All of its attributes (that you want to use in DataJoint)
        must have type hints."""

        other: "OtherStuff" = None

        x: int = 3
        """type: int"""

        y: str = "something"
        """type: str"""

    # Set DataJoint definitions in each class based on inspection
    # of class attributes.
    set_dj_definition(Stuff)

    # Schematize each of these classes (will include foreign keys)
    from pyrfume.odorants import Molecule, Vendor, ChemicalOrder

    for cls in [Molecule, Vendor, ChemicalOrder]:
        locals()[cls.__name__] = djt.schematize(cls, schema)

    # May require a Jupyter notebook or other canvas
    dj.ERD(schema).draw()
