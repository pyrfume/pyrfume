""" Definitions of datajoint tables and functions for operating the datajoint schema."""

import datajoint as dj
from datajoint.user_tables import UserTable
import quantities as pq
from pyrfume import read_config, write_config, init_config
import inspect
import warnings
from typing import Dict

dj.errors._switch_adapted_types(True)
schema = None


def set_schema_name(new_schema_name: str) -> None:
    """ Set a new schema name into the config file. The data saved 
        into the old schema will be keeped on the server unless 
        `drop_schema` was called.

    Args:
        new_schema_name (str): The new name of the schema.
    """
    init_config(False)
    write_config("DATABASE", "schema_name", new_schema_name)


def init_schema() -> None:
    """ Get the name of schema from the config file 
        and initialize the schema with it.
    """

    schema_name = read_config("DATABASE", "schema_name")
    if schema_name == "":
        warnings.warn("Schema name was not set in pyrfume config.")
        return
    context = globals()
    global schema
    schema = dj.schema(schema_name, context)
    schema.context.update({'quantity_adapter': QuantityAdapter()})


def drop_schema(force=False) -> None:
    """Drop the datajoint schema.

    Args:
        force (bool, optional): Dropping it without a pop up confirmation. 
        Defaults to False.
    """

    global schema
    if not schema:
        init_schema()
    schema.drop(force=force)
    schema = None


def get_table(table_name: str) -> UserTable:
    """ Get a single datajoint table by table name.

    Args:
        table_name (str): The name of the datajoint table.

    Returns:
        UserTable: The datajoint table.
    """
    global schema
    if not schema:
        init_schema()

    result = None
    if table_name:
        try:
            cls = globals[table_name]
            if inspect.isclass(cls):
                result = schema(cls)
        except KeyError:
            print("Class does not exist.")

    return result


def get_tables() -> Dict[str, UserTable]:
    """ Get all datajoint tables in this module.

    Returns:
        Dict[str, UserTable]: The dict that contains all the table classes.
    """

    global schema
    if not schema:
        init_schema()

    result = {}
    for key, value in globals().items():
        if inspect.isclass(value) and value is not QuantityAdapter and value is not UserTable:
            result[key] = (schema(value))

    return result


class QuantityAdapter(dj.AttributeAdapter):
    """ The datajoint adapter class that puts and gets Python Quantity objects 
        to and from the datajoint database server.
    """
    attribute_type = 'float'

    def put(self, obj: pq.Quantity):
        assert isinstance(obj, pq.Quantity)
        obj = obj.rescale(pq.mL)
        return obj.item()

    def get(self, value: float):
        return value * pq.mL


class Molecule(dj.Manual):
    definition = '''
    smiles : varchar(25)
    ---
    inchi = NULL : varchar(128)
    inchikey = "" : varchar(256)
    pubchem_id = NULL : int
    name = "" : varchar(128)
    iupac = "" : varchar(128)
    '''


class Vendor(dj.Manual):
    definition = '''
    name = "" : varchar(256)
    ---
    '''


class Product(dj.Manual):
    definition = '''
    -> Vendor 
    catalog = 0 : int
    ---
    -> Molecule
    purity = "" : varchar(64)
    batch = "" : varchar(64)
    '''


class Compound(dj.Manual):
    definition = '''
    -> Product
    date_delivered : datetime
    location = "" : varchar(64)
    ---
    date_opened = NULL : datetime
    '''


class Solution(dj.Manual):
    definition = '''
    solution_id: int auto_increment
    ---
    diution = NULL : int
    concentration = NULL : float
    value = NULL : <quantity_adapter>
    mixing_data = NULL : date
    '''

    class Compounds(dj.Part):
        definition = '''
        -> Solution
        -> Compound
        ---
        '''


class Vessel(dj.Manual):
    definition = '''
    name = "" : varchar(64)
    height = 0 : float
    base_area = "" : varchar(64)
    ---
    '''


class Odorant(dj.Manual):
    definition = '''
    odorant_id: int auto_increment
    ---
    -> Vessel
    date_prepared = NULL : datetime
    '''

    class Solutions(dj.Part):
        definition = '''
        -> Odorant
        -> Solution
        ---
        '''


class Route(dj.Manual):
    definition = '''
    route_id: int auto_increment
    ---
    name = "" : varchar(64)
    '''


class Stimulus(dj.Manual):
    definition = '''
    stimulus_id: int auto_increment
    ---
    -> Route
    '''

    class Odorants(dj.Part):
        definition = '''
        -> Stimulus
        -> Odorant
        ---
        '''


class Subject(dj.Manual):
    definition = '''
    subject_id: int auto_increment
    ---
    age : tinyint
    gender : tinyint
    detail_info : varchar(512)
    '''


class Trial(dj.Manual):
    definition = '''
    trial_id: int auto_increment
    ---
    -> Stimulus
    -> Subject
    time : timestamp
    '''


class Site(dj.Manual):
    definition = '''
    site_id: int auto_increment
    ---
    name : varchar(64)
    kind : varchar(16)
    '''


class Investigator(dj.Manual):
    definition = '''
    investigator_id: int auto_increment
    ---
    first_name : varchar(64)
    last_name : varchar(64)
    -> Site
    '''


class Technician(dj.Manual):
    definition = '''
    technician_id: int auto_increment
    ---
    first_name : varchar(64)
    last_name : varchar(64)
    -> Investigator
    '''


class Publication(dj.Manual):
    definition = '''
    publication_id: int auto_increment
    ---
    name : varchar(512)
    kind : varchar(32)
    -> Investigator
    '''


class Report(dj.Manual):
    definition = '''
    report_id: int auto_increment
    ---
    title : varchar(512)
    year : smallint
    -> Publication
    doi : varchar(128)
    
    last_name : varchar(64)
    -> Investigator
    '''


class Design(dj.Manual):
    definition = '''
    design_id: int auto_increment
    ---
    name : varchar(64)
    '''


class Block(dj.Manual):
    definition = '''
    block_id: int auto_increment
    ---
    -> Technician
    -> Design
    '''

    class Trials(dj.Part):
        definition = '''
        -> Block
        -> Trial
        ---
        '''


class Experiment(dj.Manual):
    definition = '''
    experiment_id: int auto_increment
    ---
    -> Investigator
    '''

    class Blocks(dj.Part):
        definition = '''
        -> Experiment
        -> Block
        ---
        '''


class Summary(dj.Manual):
    definition = '''
    summary_id: int auto_increment
    ---
    -> Publication
    -> Design
    '''

    class Odorants(dj.Part):
        definition = '''
        -> Summary
        -> Odorant
        ---
        '''
