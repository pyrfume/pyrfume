import math
import random
import contextlib
import io
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import multiprocessing
import dask.bag as db
from deap import base, creator, tools, algorithms


@contextlib.contextmanager
def suppress_stdout(on):
    if on:
        sys.stdout = io.StringIO()
    yield
    sys.stdout = sys.__stdout__
    
    
class OdorantSetOptimizer:
    def __init__(self, library: pd.DataFrame, n_desired: int, weights: list,
                 keep_cids: bool = None, fitness = None,
                 n_gen:int = 300, mu: int = 100, lamda:int = 200, p_cx: float=0.4, p_mut: float=0.4, sel='selBest', 
                 standardize_weights=False, npartitions=1):
        """
        params:
            library (pd.DataFrame): A pandas dataframe containing odorants and their
                                    attributes. The index should contain chemical identifiers
                                    (e.g. CIDs). The column names should be attributes,
                                     e.g. molecular weight, price, etc.  They can also be
                                      descriptors.
            n_desired (int): The size of the set of odorants desired from this
                       library.
            keep_cids (list): An iterable of CIDs that must exist in every candidate set.
                  Obviously this must be less than `n_desired`.
            weights (list): A list of 3-tuples each containing the name of a weight,
                     a method to apply to the library dataframe, and a weight
                     to apply to the function's output.
            fitness: A fitness class from DEAP
            mu (int): Size of the population (number of odorant sets under consideration
                in each generation).
            lamda (int): ...
            p_cx (float): Crossover probability in each generation.
            p_mut (float): Mutation probability in each generation.
            sel: Selection algorithm, e.g. 'selBest' or 'selNSGA2'
            standardize_weights (bool): Interpret weights as multiples of stdev of a weight function
        """

        # Turn all constructor arguments into attributes
        for key, value in locals().items():
            setattr(self, key, value)

        if self.fitness is None:
            self.fitness = BetterFitness

        weight_names, weight_functions, weight_values = zip(*self.weights)
        assert(len(weight_names) == len(set(weight_names))), "Weight names must all be unique."

        # Number of available odorants
        self.library_size = self.library.shape[0]
        
        if self.keep_cids:
            # Get library integer indices of the CIDs to keep
            self.keep = set(np.flatnonzero(self.library.index.isin(self.keep_cids)))
            self.n_needed = self.n_desired - len(self.keep)

        creator.create("Fitness", self.fitness, weights=weight_values)
        creator.create("Individual", set, fitness=creator.Fitness)

        self.toolbox = base.Toolbox()

        # Not sure what this does, but I think it says that item
        # attributes are random?
        self.toolbox.register("random_set", random.sample,
                              range(self.library_size), self.n_desired)

        # Says that an individual has something to do with attr_item,
        # and has initial size INITIAL_SET_SIZE
        self.toolbox.register("individual", tools.initIterate,
                              creator.Individual, self.toolbox.random_set)

        # Says that the population is just a list of individuals
        self.toolbox.register("population", tools.initRepeat, list,
                              self.toolbox.individual)
        self.toolbox.register("evaluate", self.eval_individual)
        self.toolbox.register("mate", self.crossover)
        self.toolbox.register("mutate", self.mutate)
        selection_algorithm = getattr(tools, sel)
        self.toolbox.register("select", selection_algorithm)
        #pool = multiprocessing.Pool()
        #self.toolbox.register("map", pool.map)
        if npartitions > 1:
            self.toolbox.register("map", self.dask_map)
        
        if self.standardize_weights:
            self.compute_weight_stats()
        
    def compute_weight_stats(self):
        """Compute weight stats for random populations to use for standardizing of fitness"""
        n_iter = 100
        weight_names = [x[0] for x in self.weights]
        
        fitnesses = pd.DataFrame(index=range(n_iter), columns=weight_names)
        for i in range(n_iter):
            ind = np.random.choice(range(self.library_size), self.n_desired, replace=False)
            fitnesses.loc[i, :] = self.eval_individual(ind, standardize=False)
        self.stds = fitnesses.std()
        self.means = fitnesses.mean()
        
    def eval_individual(self, individual, standardize=None):
        """Evaluate the fitness of the odorant set"""
        fitness = []
        for weight_name, func_name, value in self.weights:
            if isinstance(func_name, str):
                f = getattr(self, 'eval_%s' % func_name)
                component = f(individual, weight_name)
            elif isinstance(func_name, (tuple, list, set)):
                f, *args = func_name
                component = f(individual, *args)
            else:
                f = func_name
                component = f(individual)
            if self.standardize_weights and (standardize is not False):
                component = (component - self.means[weight_name])/self.stds[weight_name]
            fitness.append(component)
        return fitness

    def crossover(self, ind1, ind2):
        """Apply a crossover operation on input sets."""
        union = ind1 | ind2
        x1 = random.sample(union, self.n_desired)
        x2 = random.sample(union, self.n_desired)
        ind1.clear()
        ind1.update(x1)
        ind2.clear()
        ind2.update(x2)
        assert len(ind1) == self.n_desired, "Individual smaller than desired size."
        assert len(ind2) == self.n_desired, "Individual smaller than desired size."
        if self.keep_cids is not None:
            ind1 = self.force_keep(ind1)
            ind2 = self.force_keep(ind2)
        return ind1, ind2

    def mutate(self, individual):
        """Mutation that pops or adds some elements."""
        N_MUTATIONS = 1
        available = list(set(range(self.library_size)).difference(individual))
        to_remove = random.sample(list(individual), N_MUTATIONS)
        to_add = random.sample(available, N_MUTATIONS)
        # Removes elements found in `to_remove`
        # XOR works here to remove `to_remove` because it can
        # only contain current members of the individial
        individual ^= set(to_remove)
        # Adds elements found in `to_add`
        individual |= set(to_add)
        assert len(individual) == self.n_desired, "Individual smaller than desired size."
        if self.keep_cids is not None:
            individual = self.force_keep(individual)
        return individual,
    
    def force_keep(self, ind):
        """Ensure that the CIDs in `self.keep` get included"""
        # Consider only molecules for this individual which are not in the keep set
        candidates = ind.difference(self.keep)
        # Keep only enough of those that there are room for
        to_add = set(random.sample(candidates, self.n_needed))
        # Set this individual to contain this subset of the candidates plus the keep set
        ind.clear()
        ind.update(to_add | self.keep)
        return ind
        
    def run(self, pop=None, hof=None, quiet=False):
        self.pop = pop if pop else self.toolbox.population(n=self.mu)
        if hof is None:
            self.hof = tools.HallOfFame(self.n_gen)
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", lambda x: np.mean(x, axis=0))
        # stats.register("avg", lambda x: np.mean(x, axis=0))
        # stats.register("std", lambda x: np.std(x, axis=0).round(1))
        # stats.register("min", lambda x: np.min(x, axis=0).round(1))
        # stats.register("max", lambda x: np.max(x, axis=0).round(1))

        def best(x):
            weight_names, weight_functions, weight_values = zip(*self.weights)
            scores = [np.dot(self.hof.keys[i].values, weight_values)
                      for i in range(len(self.hof))]
            i = np.argmax(scores)
            return np.array([scores[i]] + list(self.hof.keys[i].values))

        self.stats.register("best", best)

        f = algorithms.eaMuPlusLambda
        with suppress_stdout(quiet):
            self.pop, self.logbook = f(self.pop, self.toolbox, self.mu,
                                       self.lamda, self.p_cx, self.p_mut,
                                       self.n_gen, self.stats, halloffame=self.hof)

        return self.pop, self.stats, self.hof, self.logbook

    def eval_mean(self, individual, column):
        return self.library.iloc[list(individual)][column].mean()

    def eval_sum(self, individual, column):
        return self.library.iloc[list(individual)][column].sum()
    
    def dask_map(self, f, x):
        x = db.from_sequence(x, npartitions=self.npartitions) 
        return db.map(f, x).compute()
    
    def plot_score_history(self):
        fig, axes = plt.subplots(2, math.ceil((len(self.weights)+1)/2), figsize=(20,8))
        scores = [x['best'][0] for x in self.logbook]
        axes[0, 0].plot(scores)
        axes[0, 0].set_title('Total')
        for j, (feature, stat, weight) in enumerate(self.weights):
            ax = axes.flat[j+1]
            history = [x['best'][j+1] for x in self.logbook]
            ax.plot(history)
            ax.set_title(feature)
        axes[0, 0].set_xlabel('Generation')
        plt.tight_layout()


class BetterFitness(base.Fitness):
    def __le__(self, other):
        return sum(self.wvalues) <= sum(other.wvalues)

    def __lt__(self, other):
        return sum(self.wvalues) < sum(other.wvalues)

    
def get_coverage(odorant_indices, space, sigma=2):
    """This function will be used to compute coverage of odorant space during optimization.
    We want non-selected odorants to be 'covered' as much as possible by selected ones"""  
    ind = list(odorant_indices)
    return covered[space].iloc[ind].max(axis=0).mean()
    
def get_entropy(odorant_indices, space, bins_per_dim=10):
    from scipy.stats import entropy
    """These function will be used to determine entropy of the selected odorants in the low-d manifold.
    We want the selected odorants to smoothly cover the available odorant space"""
    ind = list(odorant_indices)
    # Determine optimal bins in each dimension
    bins = {dim: np.histogram_bin_edges(embedded_coords[space][dim], bins=bins_per_dim) for dim in ('X', 'Y')}
    # Get the XY coordinates of the selected odorants in the embedding
    XY = embedded_coords_available[space].iloc[ind].values
    # Compute a 2D histogram of selected odorant counts in each bin
    histXY = np.histogram2d(*np.array(XY).T, bins=(bins['X'], bins['Y']))[0]
    # Compute the entropy of this histogram
    # Higher is flatter (less clustered)
    h = entropy(histXY.ravel())
    return h

def get_spacing(odorant_indices, space, n=5):
    """These function will be used to determine the spacing between the selected odorants in the low-d manifold.
    We don't want to select odorants which are right next to each other in odorant space"""
    ind = list(odorant_indices)
    # Pairwise distances between selected odorants
    x = embedded_distances_available2[space].iloc[ind, ind].values
    # Unraveled (ignoring diagonal)
    x = x[np.triu_indices(len(ind), 1)]
    # Penalize the closest n odorant pairs
    return np.sort(x)[:n].mean()


if __name__ == '__main__':
    np.set_printoptions(precision=2, suppress=True)
    library = pd.read_csv('data/Mainland Odor Cabinet with CIDs.csv')
    weights = [('$/mol', 'mean', 1),
               ('MW', 'sum', 2),
               ('D g/ml', 'mean', -3)]
    optimizer = OdorantSetOptimizer(library, 25, weights, n_gen=25)
    pop, stats, hof, logbook = optimizer.run()