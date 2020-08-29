import configparser
import json
import logging
import pickle
import urllib
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm, trange

from .base import CONFIG_PATH, DEFAULT_DATA_PATH
from typing import Any, List

logger = logging.getLogger("pyrfume")


def init_config(overwrite=False):
    if overwrite or not CONFIG_PATH.exists():
        config = configparser.ConfigParser()
        config["PATHS"] = {"pyrfume-data": str(DEFAULT_DATA_PATH)}
        with open(CONFIG_PATH, "w") as f:
            config.write(f)


def reset_config():
    init_config(overwrite=True)


def read_config(header, key):
    config = configparser.ConfigParser()
    init_config()
    config.read(CONFIG_PATH)
    return config[header][key]


def write_config(header, key, value):
    config = configparser.ConfigParser()
    init_config()
    config.read(CONFIG_PATH)
    config[header][key] = value
    with open(CONFIG_PATH, "w") as f:
        config.write(f)


def set_data_path(path):
    path = Path(path).resolve()
    if not path.exists():
        raise Exception("Could not find path %s" % path)
    write_config("PATHS", "pyrfume-data", str(path))


def get_data_path():
    path = read_config("PATHS", "pyrfume-data")
    path = Path(path).resolve()
    if not path.exists():
        raise Exception("Could not find data path %s" % path)
    return path


def load_data(rel_path, **kwargs):
    full_path = get_data_path() / rel_path
    is_pickle = any([str(full_path).endswith(x) for x in (".pkl", ".pickle", ".p")])
    is_excel = any([str(full_path).endswith(x) for x in (".xls", ".xlsx")])
    if is_pickle:
        with open(full_path, "rb") as f:
            data = pickle.load(f)
    elif is_excel:
        data = pd.read_excel(full_path, **kwargs)
    else:
        if "index_col" not in kwargs:
            kwargs["index_col"] = 0
        data = pd.read_csv(full_path, **kwargs)
    return data


def save_data(data, rel_path, **kwargs):
    full_path = get_data_path() / rel_path
    is_pickle = any(str(full_path).endswith(x) for x in (".pkl", ".pickle", ".p"))
    is_csv = any(str(full_path).endswith(x) for x in (".csv"))
    if is_pickle:
        with open(full_path, "wb") as f:
            pickle.dump(data, f)
    elif is_csv:
        data.to_csv(full_path, **kwargs)
    else:
        raise Exception("Unsupported extension in file name %s" % full_path.name)


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


class TriangleTest(object):
    """
    One kind of experimental test, as performed by e.g. Bushdid et al.
    A 'triangle test' has three stimuli, two of which are the same two
    odorants, and is defined by those odorants.
    """

    def __init__(self, test_uid: int, odorants: list, dilution: float, correct: bool):
        """
        Tests are defined by their universal identifier (UID), the 3
        odorants used (2 should be identical), the dilution, and the
        identity of the correct response, which should be the odd-ball.
        """

        self.id = test_uid
        self.odorants = odorants
        self.dilution = dilution
        self.correct = correct

    def add_odorant(self, odorant):
        """
        Adds one odorant to this test.
        """

        self.odorants.append(odorant)

    def add_odorants(self, odorants: list):
        """
        Adds more than one odorants to this test.
        """

        self.odorants.extend(odorants)

    @property
    def double(self):
        """
        Returns the odorant present twice in this test.
        """

        for odorant in self.odorants:
            if self.odorants.count(odorant) == 2:
                return odorant
        return None

    @property
    def single(self):
        """
        Returns the odorant present once in this test.
        """

        for odorant in self.odorants:
            if self.odorants.count(odorant) == 1:
                return odorant
        return None

    @property
    def pair(self):
        """
        Returns the odorant pair in this test, with the odorant present
        twice listed first.
        """

        return (self.double, self.single)

    @property
    def N(self):
        """
        Returns the number of components in each of the odorants.
        This a single value since they should all have the same number
        of components.
        """

        return self.double.N

    @property
    def r(self):
        """
        Returns the number of component replacements (swaps) separating one of
        the odorants from the other.
        """

        return self.double.r(self.single)

    def overlap(self, percent=False):
        """
        Returns the overlap (complement of r) between the two odorants.
        Optionally returns this as a percentage of N.
        """

        return self.double.overlap(self.single, percent=percent)

    @property
    def common_components(self):
        """
        Returns a list of components common to the two odorants.
        """

        d = set(self.double.components)
        s = set(self.single.components)
        return list(s.intersection(d))

    @property
    def unique_components(self):
        """
        Returns a list of components that exactly one of the two odorants has.
        """

        d = set(self.double.components)
        s = set(self.single.components)
        return list(s.symmetric_difference(d))

    def unique_descriptors(self, source):
        """
        Given a data source, returns a list of descriptors that
        exactly one of the two odorants has.
        """

        sl = self.single.descriptor_list(source)
        dl = self.double.descriptor_list(source)
        unique = set(dl).symmetric_difference(set(sl))
        return list(unique)

    def common_descriptors(self, source):
        """
        Given a data source, returns a list of descriptors that
        are common to the two odorants.
        """

        sl = self.single.descriptor_list(source)
        dl = self.double.descriptor_list(source)
        unique = set(dl).intersection(set(sl))
        return list(unique)

    def descriptors_correlation(self, source, all_descriptors):
        """
        Given a data source, returns the correlation between the descriptors
        of the two odorants.
        """

        sv = self.single.descriptor_vector(source, all_descriptors)
        dv = self.double.descriptor_vector(source, all_descriptors)
        return np.corrcoef(sv, dv)[1][0]

    def descriptors_correlation2(self, all_descriptors):
        """
        Returns the correlation between the descriptors
        of the two odorants, combining multiple data sources.
        """

        sv = self.single.descriptor_vector2(all_descriptors)
        dv = self.double.descriptor_vector2(all_descriptors)
        return np.corrcoef(sv, dv)[1][0]

    def descriptors_difference(self, source, all_descriptors):
        """
        Given a data source, returns the absolute difference between the
        descriptors of the two odorants.
        """

        sv = self.single.descriptor_vector(source, all_descriptors)
        dv = self.double.descriptor_vector(source, all_descriptors)
        return np.abs(sv - dv)

    def n_undescribed(self, source):
        """
        Given a data source, returns the number of components from among the
        two odorants that are not described by that source.
        """

        d = self.double.n_described_components(source)
        s = self.single.n_described_components(source)
        return (self.N - d, self.N - s)

    @classmethod
    def length(cls, v):
        return np.sqrt(np.dot(v, v))

    @classmethod
    def find_angle(cls, v1, v2):
        return np.arccos(np.dot(v1, v2) / (cls.length(v1) * cls.length(v2)))

    @classmethod
    def circmean(cls, angles):
        return np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))

    def angle(self, features, weights=None, method="sum", method_param=1.0):
        angle = None
        if method == "sum":
            v1 = self.single.vector(features, weights=weights, method=method)
            v2 = self.double.vector(features, weights=weights, method=method)
            angle = self.find_angle(v1, v2)
        elif method == "nn":  # Nearest-Neighbor.
            m1 = self.single.matrix(features, weights=weights)
            m2 = self.double.matrix(features, weights=weights)
            angles = []
            for i in range(m1.shape[0]):
                angles_i = []
                for j in range(m2.shape[0]):
                    one_angle = self.find_angle(m1[i, :], m2[j, :])
                    if np.isnan(one_angle):
                        one_angle = 1.0
                    angles_i.append(one_angle)
                angles_i = np.array(sorted(angles_i))
                from scipy.stats import geom

                weights_i = geom.pmf(range(1, len(angles_i) + 1), method_param)
                angles.append(np.dot(angles_i, weights_i))
            angle = np.abs(angles).mean()  # circmean(angles)
        return angle

    def norm(self, features, order=1, weights=None, method="sum"):
        v1 = self.single.vector(features, weights=weights, method=method)
        v2 = self.double.vector(features, weights=weights, method=method)
        dv = v1 - v2
        dv = np.abs(dv) ** order
        return np.sum(dv)

    def distance(self, features, weights=None, method="sum"):
        v1 = self.single.vector(features, weights=weights, method=method)
        v2 = self.double.vector(features, weights=weights, method=method)
        return np.sqrt(((v1 - v2) ** 2).sum())

    def fraction_correct(self, results: list):
        num, denom = 0.0, 0.0
        for result in results:
            if result.test.id == self.id:
                num += result.correct
                denom += 1
        return num / denom


class Result(object):
    """
    A test result, corresponding to one test given to one subject.
    """

    def __init__(self, test: TriangleTest, subject_id: int, correct: bool):
        """
        Results are defined by the test to which they correspond,
        the id of the subject taking that test, and whether the subject
        gave the correct answer.
        """

        self.test = test
        self.subject_id = subject_id
        self.correct = correct


class Distance(object):
    """
    An odorant distance, corresponding to distance between two odorants.
    No particular implementation for computing distance is mandated.
    """

    def __init__(self, odorant_i: Any, odorant_j: Any, distance: float):
        self.odorant_i = odorant_i
        self.odorant_j = odorant_j
        self.distance = distance


def odorant_distances(results, subject_id=None):
    """
    Given the test results, returns a dictionary whose keys are odorant pairs
    and whose values are psychometric distances between those pairs,
    defined as the fraction of discriminations that were incorrect.
    This can be limited to one subject indicated by subject_id, or else
    by default it pools across all subjects.
    """

    distances = {}
    distance_n_subjects = {}
    for result in results:
        if subject_id and result.subject_id != subject_id:
            continue
        pair = result.test.pair
        if pair not in distances:
            distances[pair] = 0
            distance_n_subjects[pair] = 0
        distances[pair] += 0.0 if result.correct else 1.0
        distance_n_subjects[pair] += 1
    for pair in list(distances.keys()):
        # Divided by the total number of subjects.
        distances[pair] /= distance_n_subjects[pair]
    return distances


def ROC(results, N):
    """
    Given test results and a number of components N, returns a distribution
    of the number of distinct components 'r' for correct trials (right) and
    incorrect trials (wrong), in tests using odorants with N total components.
    These can later be plotted or used to generated an ROC curve.
    """

    right = []
    wrong = []
    for result in results:
        if result.test.N == N:
            r = result.test.r
            if result.correct:
                right.append(r)
            else:
                wrong.append(r)
    right = np.array(right)  # Distribution of r for correct trials.
    wrong = np.array(wrong)  # Distribution of r for incorrect trials.
    return (right, wrong)


def correct_matrix(results, N, overlap):
    """
    Given test results, a number of components N, and a level of overlap
    between odorants, returns a num_subjects by num_test matrix of booleans
    corresponding to the correctness of that subject's response on that test.
    """

    results = [
        r
        for r in results
        if (N is None or r.test.N == N) and (overlap is None or r.test.overlap() == overlap)
    ]
    subjects = [r.subject_id for r in results]
    subjects = list(set(subjects))
    tests = [r.test for r in results]
    tests = list(set(tests))
    correct = np.zeros((len(subjects), len(tests)))
    correct -= 1  # Set to make sure every point gets set to 0 or 1 later.
    for result in results:
        i = subjects.index(result.subject_id)
        j = tests.index(result.test)
        correct[i, j] = result.correct
    return correct, subjects, tests
