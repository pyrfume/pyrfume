import json
import numpy as np
import urllib


class Mixture(object):
    """
    A mixture of molecules, defined by the presence of absence of the
    candidate molecules in the mixture.
    """

    def __init__(self, C: int, components: list=None):
        """
        Builds odorant from a list of components.
        """
        self.C = C
        self.components = components if components else []

    name = None  # Name of odorant, built from a hash of component names.

    C = None  # Number of components from which to choose.

    def components_vector(self, all_components: list=None, normalize: float=0):

        vector = np.zeros(self.C)
        for component in self.components:
            vector[all_components.index(component)] = 1
        if normalize:
            denom = (np.abs(vector) ** normalize).sum()
            vector /= denom
        return vector

    @property
    def N(self):
        """
        Number of components in this odorant.
        """

        return len(self.components)

    def r(self, other):
        """
        Number of replacements (swaps) to get from self to another odorant.
        """

        if len(self.components) == len(other.components):
            return self.hamming(other) / 2
        else:
            return None

    def overlap(self, other, percent=False):
        """
        Overlap between self and another odorant.  Complement of r.
        Optionally report result as percent relative to number of components.
        """

        overlap = self.N - self.r(other)
        if percent:
            overlap = overlap * 100.0 / self.N
        return overlap

    def hamming(self, other):
        """
        Hamming distance between self and another odorant.
        Synonymous with number of d, the number of total 'moves' to go from
        one odorant to another.
        """

        x = set(self.components)
        y = set(other.components)
        diff = len(x) + len(y) - 2 * len(x.intersection(y))
        return diff

    def add_component(self, component):
        """
        Adds one component to an odorant.
        """

        self.components.append(component)

    def remove_component(self, component):
        """
        Removes one component to an odorant.
        """

        self.components.remove(component)

    def descriptor_list(self, source):
        """
        Given a data source, returns a list of descriptors about this odorant.
        """

        descriptors = []
        for component in self.components:
            if source in component.descriptors:
                desc = component.descriptors[source]
                if type(desc) == list:
                    descriptors += desc
                if type(desc) == dict:
                    descriptors += [key for key, value in list(desc.items()) if value > 0.0]
        return list(set(descriptors))  # Remove duplicates.

    def descriptor_vector(self, source, all_descriptors):
        """
        Given a data source, returns a vector of descriptors about this
        odorant. The vector will contain positive floats.
        """

        vector = np.zeros(len(all_descriptors[source]))
        for component in self.components:
            if source in component.descriptors:
                desc = component.descriptors[source]
                if type(desc) == list:
                    for descriptor in desc:
                        index = all_descriptors[source].index(descriptor)
                        assert index >= 0
                        vector[index] += 1
                if type(desc) == dict:
                    this_vector = np.array([value for key, value in sorted(desc.items())])
                    vector += this_vector
        return vector

    def descriptor_vector2(self, all_descriptors):
        """
        Returns a vector of descriptors about this odorant, combining multiple
        data sources.
        """

        n_descriptors_dravnieks = len(all_descriptors["dravnieks"])
        n_descriptors_sigma_ff = len(all_descriptors["sigma_ff"])
        vector = np.zeros(n_descriptors_dravnieks + n_descriptors_sigma_ff)
        for component in self.components:
            if "dravnieks" in component.descriptors:
                desc = component.descriptors["dravnieks"]
                this_vector = np.array([value for key, value in sorted(desc.items())])
                vector[0:n_descriptors_dravnieks] += this_vector
            elif "sigma_ff" in component.descriptors:
                desc = component.descriptors["sigma_ff"]
                for descriptor in desc:
                    index = all_descriptors["sigma_ff"].index(descriptor)
                    assert index >= 0
                    vector[n_descriptors_dravnieks + index] += 1
        return vector

    def described_components(self, source):
        """
        Given a data source, returns a list of the components which are
        described by that source, i.e. those that have descriptors.
        """

        return [component for component in self.components if source in component.descriptors]

    def n_described_components(self, source):
        """
        Given a data source, returns the number of components that are
        described by that data source.
        """

        return len(self.described_components(source))

    def fraction_components_described(self, source):
        """
        Given a data source, returns the fraction of components that are
        described by that data source.
        """

        return self.n_described_components(source) / self.N

    def matrix(self, features, weights=None):
        matrix = np.vstack(
            [
                component.vector(features, weights=weights)
                for component in self.components
                if component.cid in features
            ]
        )
        if 0:  # matrix.shape[0] != self.N:
            print(
                ("Mixture has %d components but only " "%d vectors were computed")
                % (self.N, matrix.shape[0])
            )
        return matrix

    def vector(self, features, weights=None, method="sum"):
        matrix = self.matrix(features, weights=weights)
        if method == "sum":
            vector = matrix.sum(axis=0)
        else:
            vector = None
        return vector

    def __str__(self):
        """
        String representation of the odorant.
        """

        return ",".join([str(x) for x in self.components])


class Component(object):
    """
    A single molecule, which may or may not be present in an odorant.
    """

    def __init__(self, component_id: int, name: str, cas: str, percent: float, solvent: "Compound"):
        """
        Components are defined by a component_id from the Bushdid et al
        supplemental material, a name, a CAS number, a percent dilution,
        and a solvent.
        """

        self.id = component_id
        self.name = name
        self.cas = cas
        self.cid_ = None
        self.percent = percent
        self.solvent = solvent
        self.descriptors = {}  # An empty dictionary.

    @property
    def cid(self):
        cid = None
        if self.cid_:
            cid = self.cid_
        else:
            url_template = "https://pubchem.ncbi.nlm.nih.gov/" "rest/pug/compound/name/%s/cids/JSON"
            for query in self.cas, self.name:
                try:
                    url = url_template % query
                    page = urllib.request.urlopen(url)
                    string = page.read().decode("utf-8")
                    json_data = json.loads(string)
                    cid = json_data["IdentifierList"]["CID"][0]
                except urllib.error.HTTPError:
                    print(query)
                else:
                    break
            self.cid_ = cid
        return cid

    def set_descriptors(self, source: str, cas_descriptors) -> None:
        """
        Given a data source, sets descriptors for this odorant using
        a dictionary where CAS numbers are keys, and descriptors are values.
        """

        assert type(source) == str and len(source)
        if self.cas in cas_descriptors:
            self.descriptors[source] = cas_descriptors[self.cas]
            # For sigma_ff this will be a list.
            # For dravnieks this will be a dict.

    def vector(self, features, weights=None):
        if self.cid in features:
            feature_values = np.array(list(features[self.cid].values()))
            if weights is None:
                weights = np.ones(feature_values.shape)
            result = feature_values * weights
        else:
            result = None
        return result

    def __str__(self):
        return self.name
