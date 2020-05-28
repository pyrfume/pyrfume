import networkx as nx
from rdkit.Chem import Atom, Bond, Mol, MolFromSmiles, RemoveHs


def smi_has_error(smi) -> bool:
    smi = smi.strip()
    n_open_parenthesis = sum(1 for c in smi if c == "(")
    n_close_parenthesis = sum(1 for c in smi if c == ")")
    n_open_parenthesis_square = sum(1 for c in smi if c == "[")
    n_close_parenthesis_square = sum(1 for c in smi if c == "]")
    return (n_open_parenthesis != n_close_parenthesis) or (
        n_open_parenthesis_square != n_close_parenthesis_square
    )


def smiles_to_eden(smiles) -> nx.Graph:
    if smi_has_error(smiles) is False:
        mol = MolFromSmiles(smiles)
        # remove hydrogens
        mol = RemoveHs(mol)
        graph = rdkit_to_networkx(mol)
        if len(graph):
            graph.graph["info"] = smiles
            return graph


def rdkit_to_networkx(mol: Mol) -> nx.Graph:
    graph = nx.Graph()
    # atoms
    atom: Atom
    for atom in mol.GetAtoms():
        node_id = atom.GetIdx()
        label = atom.GetSymbol()
        graph.add_node(node_id, label=label)
    # bonds
    bond: Bond
    for bond in mol.GetBonds():
        label = bond.GetBondType()
        graph.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), label=label)
    return graph
