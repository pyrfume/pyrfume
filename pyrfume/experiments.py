import numpy as np
from scipy.stats import geom
from typing import Any

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
