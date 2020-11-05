import datajoint as dj
import quantities as pq
from pyrfume import read_config

dj.errors._switch_adapted_types(True)

schema_name = read_config("DATABASE", "schema_name")
context = locals()
schema = dj.schema(schema_name, context)


class QuantityAdapter(dj.AttributeAdapter):
    attribute_type = 'float'

    def put(self, obj: pq.Quantity):
        assert isinstance(obj, pq.Quantity)
        obj = obj.rescale(pq.mL)
        return obj.item()

    def get(self, value: float):
        return value * pq.mL

schema.context.update({'quantity_adapter': QuantityAdapter()})

@schema
class Molecule(dj.Manual):
    definition = '''
    smiles : varchar(25)
    ---
    inchi = NULL : int
    inchikey = "" : varchar(256)
    pubchem_id = "" : varchar(256)
    name = "" : varchar(256)
    iupac = "" : varchar(256)
    '''


@schema
class Vendor(dj.Manual):
    definition = '''
    name = "" : varchar(256)
    ---
    '''

@schema
class Product(dj.Manual):
    definition = '''
    -> Vendor 
    catalog = 0 : int
    ---
    -> Molecule
    purity = "" : varchar(64)
    batch = "" : varchar(64)
    '''

@schema
class Compound(dj.Manual):
    definition = '''
    -> Product
    date_delivered = NULL : datetime
    location = "" : varchar(64)
    ---
    date_opened = NULL : datetime
    '''

@schema
class Solution(dj.Manual):
    definition = '''
    solution_id: int auto_increment
    ---
    diution = NULL : int
    concentration = NULL : float
    value = NULL : <quantity_adapter>
    data = NULL : varchar(4096)
    '''
    class Compounds(dj.Part):
        definition = '''
        -> Solution
        -> Compound
        ---
        '''

@schema
class Vessel(dj.Manual):
    definition = '''
    name = "" : varchar(64)
    height = NULL : float
    base_area = "" : varchar(64)
    ---
    '''

@schema
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

@schema
class Route(dj.Manual):
    definition = '''
    route_id: int auto_increment
    ---
    name = "" : varchar(64)
    '''

@schema
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
