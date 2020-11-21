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
    pubchem_id = "" : varchar(32)
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
    date_delivered : datetime
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
    mixing_data = NULL : date
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
    height = 0 : float
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

@schema
class Subject(dj.Manual):
    definition = '''
    subject_id: int auto_increment
    ---
    -> age : tinyint
    -> gender : tinyint
    -> detail_info : varchar(65535)
    '''

@schema
class Trial(dj.Manual):
    definition = '''
    trial_id: int auto_increment
    ---
    -> Stimulus
    -> Subject
    -> time : timestamp
    '''

@schema
class Site(dj.Manual):
    definition = '''
    site_id: int auto_increment
    ---
    -> name : varchar(64)
    -> kind : varchar(16)
    '''

@schema
class Investigator(dj.Manual):
    definition = '''
    investigator_id: int auto_increment
    ---
    -> first_name : varchar(64)
    -> last_name : varchar(64)
    -> Site
    '''

@schema
class Technician(dj.Manual):
    definition = '''
    technician_id: int auto_increment
    ---
    -> first_name : varchar(64)
    -> last_name : varchar(64)
    -> Investigator
    '''

@schema
class Publication(dj.Manual):
    definition = '''
    publication_id: int auto_increment
    ---
    -> name : varchar(1024)
    -> kind : varchar(32)
    -> Investigator
    '''

@schema
class Report(dj.Manual):
    definition = '''
    report_id: int auto_increment
    ---
    -> title : varchar(1024)
    -> year : smallint
    -> Publication
    -> doi : varchar(128)
    
    -> last_name : varchar(64)
    -> Investigator
    '''

@schema
class Design(dj.Manual):
    definition = '''
    design_id: int auto_increment
    ---
    -> name : varchar(64)
    '''

@schema
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

@schema
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

@schema
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
