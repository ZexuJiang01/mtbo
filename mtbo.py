import gpytorch
import numpy as np
import GPyOpt
import copy
import numpy as np
import array
import random
import matplotlib.pyplot as plt
import torch

from gpytorch.kernels import ScaleKernel
from gpytorch.kernels import MaternKernel
from scipy.stats import norm
from deap import base
from deap.benchmarks.tools import hypervolume
from deap import creator
from deap import tools
from deap import algorithms
import datetime
from pyDOE2 import lhs
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans

'''
multi object functions
'''


def zdt1(x: np.ndarray) -> np.ndarray:
    """
    Args:
        x (np.ndarray): x must be in [0, 1]
    Returns:
        multi_objective_value (np.ndarray):
    """
    if type(x) is not np.ndarray:
        x_in = np.array(x)
    else:
        x_in = copy.deepcopy(x)
    if np.any(x_in > 1) or np.any(x_in < 0):
        raise ValueError('0 <= x <= 1 is required')
    n_dv = x_in.shape[0]
    x_in = x_in.T
    
    
    g = 1 + 9 * sum(x_in[1:]) / (n_dv - 1)
    h = 1 - np.sqrt(x_in[0] / g)
    f1 = x_in[0]
    f2 = g * h
    return np.array([f1, f2])


def zdt2(x: np.ndarray) -> np.ndarray:
    """
    Args:
        x (np.ndarray): x must be in [0, 1]
    Returns:
        multi_objective_value (np.ndarray):
    """
    if type(x) is not np.ndarray:
        x_in = np.array(x)
    else:
        x_in = copy.deepcopy(x)
    if np.any(x_in > 1) or np.any(x_in < 0):
        raise ValueError('0 <= x <= 1 is required')
    n_dv = x_in.shape[0]
    x_in = x_in.T

    
    g = 1 + 9 * sum(x_in[1:]) / (n_dv - 1)
    h = 1 - (x_in[0] / g) ** 2
    f1 = x_in[0]
    f2 = g * h
    return np.array([f1, f2])

def zdt3(x: np.ndarray) -> np.ndarray:
    """
    Args:
        x (np.ndarray): x must be in [0, 1]
    Returns:
        multi_objective_value (np.ndarray):
    """
    if type(x) is not np.ndarray:
        x_in = np.array(x)
    else:
        x_in = copy.deepcopy(x)
    if np.any(x_in > 1) or np.any(x_in < 0):
        raise ValueError('0 <= x <= 1 is required')
    n_dv = x_in.shape[0]
    x_in = x_in.T
    
    
    g = 1 + 9 * sum(x_in[1:]) / (n_dv - 1)
    h = 1 - np.sqrt(x_in[0] / g) - x_in[0]/g * np.sin(10 * np.pi * x_in[0])
    f1 = x_in[0]
    f2 = g * h
    return np.array([f1, f2])

def zdt4(x: np.ndarray) -> np.ndarray:
    """
    Args:
        x (np.ndarray): x must be in [0, 1]
    Returns:
        multi_objective_value (np.ndarray):
    """
    if type(x) is not np.ndarray:
        x_in = np.array(x)
    else:
        x_in = copy.deepcopy(x)
    if np.any(x_in > 1) or np.any(x_in < 0):
        raise ValueError('0 <= x <= 1 is required')
    n_dv = x_in.shape[0]
    x_in = x_in.T
    
    
    g = 1 + 10 * (n_dv-1) * sum(np.sqrt(x_in[1:]) - 10 * np.cos(4 * np.pi * x_in[1:]))
    h = 1 - np.sqrt(x_in[0] / g)
    f1 = x_in[0]
    f2 = g * h
    return np.array([f1, f2])

def schaffer(x: np.ndarray) -> np.ndarray:
    """
    Args:
        x (np.ndarray): x must be in [0, 1]
    Returns:
        multi_objective_value (np.ndarray):
    """
    if type(x) is not np.ndarray:
        x_in = np.array(x)
    else:
        x_in = copy.deepcopy(x)
    if np.any(x_in > 1) or np.any(x_in < 0):
        raise ValueError('0 <= x <= 1 is required')
    x_in = x_in.T

    f1 = x_in[0]**2
    f2 = (x_in[0]-2)**2
    return np.array([f1, f2])


def binh_korn(x: np.ndarray) -> np.ndarray:
    """
    Args:
        x (np.ndarray): x must be in [0, 1]
    Returns:
        multi_objective_value (np.ndarray):
    """
    if type(x) is not np.ndarray:
        x_in = np.array(x)
    else:
        x_in = copy.deepcopy(x)
    if np.any(x_in > 1) or np.any(x_in < 0):
        raise ValueError('0 <= x <= 1 is required')

    #x_in = copy.deepcopy(x)
    x_in = x_in.T
    x_in[0] = x_in[0] * 5
    x_in[1] = x_in[1] * 3

    f1 = 4 * (x_in[0] ** 2) + 4 * (x_in[1] ** 2)
    f2 = (x_in[0] - 5) ** 2 + (x_in[1] - 5) ** 2
    g1 = (x_in[0] - 5) ** 2 + x_in[1] ** 2 - 25
    g2 = 7.7 - (x_in[0] - 8) ** 2 + (x_in[1] + 3) ** 2
    return np.array([f1, f2, g1, g2])


def chakong_haimes(x: np.ndarray) -> np.ndarray:
    """
    Args:
        x (np.ndarray): x must be in [0, 1]
    Returns:
        multi_objective_value (np.ndarray):
    """
    if type(x) is not np.ndarray:
        x_in = np.array(x)
    else:
        x_in = copy.deepcopy(x)
    if np.any(x_in > 1) or np.any(x_in < 0):
        raise ValueError('0 <= x <= 1 is required')
    #x_in = copy.deepcopy(x)
    x_in = x_in.T
    x_in[0] = x_in[0] * 40 - 20
    x_in[1] = x_in[1] * 40 - 20

    f1 = 2 + (x_in[0] - 2) ** 2 + (x_in[1] - 1) ** 2
    f2 = 9 * x_in[0] - (x_in[1] - 1) ** 2
    g1 = x_in[0] ** 2 + x_in[1] ** 2 - 255
    g2 = x_in[0] - 3 * x_in[1] + 10

    return np.array([f1, f2, g1, g2])


def osyczka_kundu(x: np.ndarray) -> np.ndarray:
    """
    Args:
        x (np.ndarray): x must be in [0, 1]
    Returns:
        multi_objective_value (np.ndarray):
    """
    if type(x) is not np.ndarray:
        x_in = np.array(x)
    else:
        x_in = copy.deepcopy(x)
    if np.any(x_in > 1) or np.any(x_in < 0):
        raise ValueError('0 <= x <= 1 is required')
    #x_in = copy.deepcopy(x)
    x_in = x_in.T
    x_in[0] = x_in[0] * 10
    x_in[1] = x_in[1] * 10
    x_in[2] = x_in[2] * 4 + 1
    x_in[3] = x_in[3] * 6
    x_in[4] = x_in[4] * 4 + 1
    x_in[5] = x_in[5] * 10

    f1 = -25 * (x_in[0] - 2) ** 2 - (x_in[1] - 2) ** 2
    - (x_in[2] - 1) ** 2 - (x_in[3] - 4) ** 2
    - (x_in[4] - 1) ** 2
    f2 = 0
    for i in range(0, 6):
        f2 = f2 + x_in[i] ** 2
    g1 = (x_in[0] + x_in[1] - 2)
    g2 = (6 - x_in[0] - x_in[1])
    g3 = (2 + x_in[1] - x_in[0])
    g4 = (2 - x_in[0] + 3 * x_in[1])
    g5 = (4 - (x_in[2] - 3) ** 2 - x_in[3])
    g6 = ((x_in[4] - 3) ** 2 + x_in[5] - 4)
    return np.array([f1, f2, g1, g2, g3, g4, g5, g6])

'''
exact gp
'''

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = ScaleKernel(
            MaternKernel(5/2, ard_num_dims=train_x.shape[1]))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

'''
acquisition function
'''

class acquisition:
    def ei(y_pred, y_train):
        mean = y_pred.mean.numpy()[0],
        std = y_pred.stddev.numpy()[0],
        y_min = y_train.numpy().min()
        z = (mean - y_min) / std
        out = (mean - y_min) * norm.cdf(z) + std * norm.pdf(z)
        return out[0]

    def ucb(y_pred, y_train):
        mean = y_pred.mean.numpy()[0],
        std = y_pred.stddev.numpy()[0],
        n_sample = y_train.numpy().shape[0]
        out = mean[0] + (np.sqrt(np.log(n_sample) / n_sample)) * std[0]

        return out

'''
multi object optimizer=NSGA2
'''


class NSGA2():
    def __init__(self,
                 evaluation_function=None,
                 bound_low=0.0,
                 bound_up=1.0,
                 n_design_variables_dimension=30,
                 n_population=16,
                 n_generation=50,
                 crossover_probability=0.9,
                 random_seed=9):
        self.random_seed = random_seed
        random.seed(self.random_seed)

        self.toolbox = base.Toolbox()
        self.evaluation_function = evaluation_function
        self.bound_low = bound_low
        self.bound_up = bound_up
        self.n_design_variables_dimension =\
            n_design_variables_dimension
        self.n_population = n_population
        self.n_generation = n_generation
        self.crossover_probability = crossover_probability

    def setup(self):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
        creator.create("Individual", array.array, typecode='d',
                       fitness=creator.FitnessMin)

        self.toolbox.register("attr_float", self.uniform,
                              self.bound_low, self.bound_up,
                              self.n_design_variables_dimension)
        self.toolbox.register("individual", tools.initIterate,
                              creator.Individual, self.toolbox.attr_float)
        self.toolbox.register("population", tools.initRepeat,
                              list, self.toolbox.individual)

        if self.evaluation_function:
            self.toolbox.register("evaluate", self.evaluation_function)

        self.toolbox.register("mate", tools.cxSimulatedBinaryBounded,
                              low=self.bound_low, up=self.bound_up, eta=20.0)
        self.toolbox.register("mutate", tools.mutPolynomialBounded,
                              low=self.bound_low, up=self.bound_up, eta=20.0,
                              indpb=1.0 / self.n_design_variables_dimension)
        self.toolbox.register("select", tools.selNSGA2)

        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean, axis=0)
        self.stats.register("std", np.std, axis=0)
        self.stats.register("min", np.min, axis=0)
        self.stats.register("max", np.max, axis=0)
        self.logbook = tools.Logbook()
        self.logbook.header = "gen", "evals", "std", "min", "avg", "max"
        self.pop = self.toolbox.population(n=self.n_population)

    def run(self):
        self.setup()
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in self.pop if not ind.fitness.valid]
        fitnesses = list(
            (self.toolbox.map(self.toolbox.evaluate, invalid_ind)))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # This is just to assign the crowding distance to the individuals
        # no actual selection is done
        self.pop = self.toolbox.select(self.pop, len(self.pop))

        record = self.stats.compile(self.pop)
        self.logbook.record(gen=0, evals=len(invalid_ind), **record)
        print(self.logbook.stream)

        # Begin the generational process
        for i_generation in range(1, self.n_generation):
            # Vary the population
            offspring = tools.selTournamentDCD(self.pop, len(self.pop))
            offspring = [self.toolbox.clone(ind) for ind in offspring]

            for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                if random.random() <= self.crossover_probability:
                    self.toolbox.mate(ind1, ind2)

                self.toolbox.mutate(ind1)
                self.toolbox.mutate(ind2)
                del ind1.fitness.values, ind2.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Select the next generation population
            self.pop = self.toolbox.select(
                self.pop + offspring, self.n_population)
            record = self.stats.compile(self.pop)
            self.logbook.record(
                gen=i_generation, evals=len(invalid_ind), **record)
            print(self.logbook.stream)

        print("Final population hypervolume is %f" %
              hypervolume(self.pop, [11.0, 11.0]))
        return self.pop, self.logbook

    def uniform(self, low, up, size=None):
        try:
            return [random.uniform(a, b) for a, b in zip(low, up)]
        except TypeError:
            return [random.uniform(a, b) for a, b in zip([low] * size,
                                                         [up] * size)]


'''
multi object optimizer=NSGA3
'''

class NSGA3():
    def __init__(self,
                 evaluation_function=None,
                 bound_low=0.0,
                 bound_up=1.0,
                 n_design_variables_dimension=30,
                 n_population=16,
                 n_generation=50,
                 crossover_probability=0.9,
                 random_seed=9):
        self.random_seed = random_seed
        random.seed(self.random_seed)

        self.toolbox = base.Toolbox()
        self.evaluation_function = evaluation_function
        self.bound_low = bound_low
        self.bound_up = bound_up
        self.n_design_variables_dimension =\
            n_design_variables_dimension
        self.n_population = n_population
        self.n_generation = n_generation
        self.crossover_probability = crossover_probability
        self.CXPB=1.0
        self.MUTPB=1.0
    
    def setup(self):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
        creator.create("Individual", array.array, typecode='d',
                       fitness=creator.FitnessMin)
        
        self.toolbox.register("attr_float", self.uniform,
                              self.bound_low, self.bound_up,
                              self.n_design_variables_dimension)
        self.toolbox.register("individual", tools.initIterate,
                              creator.Individual, self.toolbox.attr_float)
        self.toolbox.register("population", tools.initRepeat,
                              list, self.toolbox.individual)

        if self.evaluation_function:
            self.toolbox.register("evaluate", self.evaluation_function)

        self.toolbox.register("mate", tools.cxSimulatedBinaryBounded,
                              low=self.bound_low, up=self.bound_up, eta=20.0)
        self.toolbox.register("mutate", tools.mutPolynomialBounded,
                              low=self.bound_low, up=self.bound_up, eta=20.0,
                              indpb=1.0 / self.n_design_variables_dimension)
        
        ref_points = tools.uniform_reference_points(2, self.n_population)
        self.toolbox.register("select", tools.selNSGA3,ref_points=ref_points)
        
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean, axis=0)
        self.stats.register("std", np.std, axis=0)
        self.stats.register("min", np.min, axis=0)
        self.stats.register("max", np.max, axis=0)
        self.logbook = tools.Logbook()
        self.logbook.header = "gen", "evals", "std", "min", "avg", "max"
        self.pop = self.toolbox.population(n=self.n_population)
        
        
    def run(self):
        self.setup()
       
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in self.pop if not ind.fitness.valid]
        fitnesses = list(
            (self.toolbox.map(self.toolbox.evaluate, invalid_ind)))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # This is just to assign the crowding distance to the individuals
        # no actual selection is done
        self.pop = self.toolbox.select(self.pop, len(self.pop))

        record = self.stats.compile(self.pop)
        self.logbook.record(gen=0, evals=len(invalid_ind), **record)
        print(self.logbook.stream)

        # Begin the generational process
        for i_generation in range(1, self.n_generation):
            # Vary the population
            #offspring = tools.selTournamentDCD(self.pop, len(self.pop))
            #offspring = [self.toolbox.clone(ind) for ind in offspring]
            offspring = algorithms.varAnd(self.pop, self.toolbox, self.CXPB, self.MUTPB)
            '''
            for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                if random.random() <= self.crossover_probability:
                    self.toolbox.mate(ind1, ind2)

                self.toolbox.mutate(ind1)
                self.toolbox.mutate(ind2)
                del ind1.fitness.values, ind2.fitness.values
            '''
            # Evaluate the individuals with an invalid fitness
            '''
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            '''
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Select the next generation population
            self.pop = self.toolbox.select(
                self.pop + offspring, self.n_population)
            record = self.stats.compile(self.pop)
            self.logbook.record(
                gen=i_generation, evals=len(invalid_ind), **record)
            print(self.logbook.stream)

        print("Final population hypervolume is %f" %
              hypervolume(self.pop, [11.0, 11.0]))
        return self.pop, self.logbook
    
    def uniform(self, low, up, size=None):
        try:
            return [random.uniform(a, b) for a, b in zip(low, up)]
        except TypeError:
            return [random.uniform(a, b) for a, b in zip([low] * size,
                                                         [up] * size)]


'''
bayesopt
'''
import copy
import numpy as np
import torch
import gpytorch
from pyDOE2 import lhs
from sklearn.cluster import KMeans


class MultiObjectiveBayesianOpt():
    def __init__(self,
                 evaluation_function=None,
                 Initializer=lhs,
                 surrogate_model=ExactGPModel,
                 optimizer=NSGA2,
                 acquisition=acquisition.ei,
                 n_objective_dimension=2,
                 n_design_variables_dimension=30,
                 n_initial_sample=16,
                 bayesian_optimization_iter_max=10,
                 likelihood_optimization_iter_max=500,#5000
                 likelihood_optimization_criteria=1e-8,
                 n_new_samples=8,
                 n_ga_population=100,
                 n_ga_generation=500,#100
                 batch_size=10
                 ):
        self.Initializer = Initializer
        self.surrogate_model = surrogate_model
        self.model = [None] * n_objective_dimension
        self.likelihood = [None] * n_objective_dimension
        self.optimizer = optimizer
        self.evaluation_function = evaluation_function
        self.acquisition = acquisition
        self.n_objective_dimension = n_objective_dimension
        self.n_design_variables_dimension = n_design_variables_dimension
        self.n_initial_sample = n_initial_sample
        self.train_x = None
        self.train_y = [None] * n_objective_dimension
        self.new_x = None
        self.bayesian_optimization_iter_max = \
            bayesian_optimization_iter_max
        self.likelihood_optimization_iter_max = \
            likelihood_optimization_iter_max
        self.likelihood_optimization_criteria = \
            likelihood_optimization_criteria
        self.n_new_samples = n_new_samples
        self.n_ga_population = n_ga_population
        self.n_ga_generation = n_ga_generation
        self.batch_size=batch_size

    def _initialize(self):
        self.train_x = self.Initializer(
            self.n_design_variables_dimension,
            self.n_initial_sample).astype(np.float32)
        self.train_x = torch.from_numpy(self.train_x)
        self.train_y = torch.from_numpy(
            self.evaluation_function(self.train_x).T)
        return

    def _train_likelihood(self):
        
        for i_obj in range(self.n_objective_dimension):
            self.likelihood[i_obj] = \
                gpytorch.likelihoods.GaussianLikelihood()
            self.model[i_obj] = self.surrogate_model(
                self.train_x, self.train_y[:, i_obj], self.likelihood[i_obj])
            self.model[i_obj].train()
            self.likelihood[i_obj].train()

            
            # Use the adam optimizer for likelihood optimization
            optimizer_likelihood = torch.optim.Adam([
                # Includes GaussianLikelihood parameters
                {'params': self.model[i_obj].parameters()},
            ], lr=0.1)

            # "Loss" for GPs - the marginal log likelihood
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(
                self.likelihood[i_obj], self.model[i_obj])

            loss_prev = 0.1
            for i in range(self.likelihood_optimization_iter_max):
                # Zero gradients from previous iteration
                optimizer_likelihood.zero_grad()
                # Output from model
                output = self.model[i_obj](self.train_x)
                # Calc loss and backprop gradients
                loss = - \
                    mll(output, self.train_y[:, i_obj])
                loss.backward()
                loss_residual = abs(loss.item() - loss_prev) / abs(loss_prev)
                loss_prev = loss.item()
                print('Iter %d/%d - Loss: %.3f  res: %.8f' % (
                    i + 1, self.likelihood_optimization_iter_max,
                    loss.item(),
                    loss_residual
                ))
                if loss_residual < self.likelihood_optimization_criteria:
                    break
                optimizer_likelihood.step()

        return self.model

    def _wrap_model_and_acquisition(self):

        def ei_with_surrogate_model(x):
            for i_obj in range(self.n_objective_dimension):
                self.model[i_obj].eval()
                self.likelihood[i_obj].eval()
            y_pred = [None] * self.n_objective_dimension
            res = [None] * self.n_objective_dimension
            for i_obj in range(self.n_objective_dimension):
                y_pred[i_obj] = \
                    self.likelihood[i_obj](
                        self.model[i_obj](torch.tensor([x])))
                res[i_obj] = self.acquisition(y_pred[i_obj],
                                              self.train_y[:, i_obj])
            return res
        return ei_with_surrogate_model

    def _find_new_sample(self):
        for i_obj in range(self.n_objective_dimension):
            self.model[i_obj].eval()
            self.likelihood[i_obj].eval()
        with torch.no_grad():
            ei_with_surrogate_model = self._wrap_model_and_acquisition()
            opt = copy.deepcopy(self.optimizer(
                evaluation_function=ei_with_surrogate_model,
                n_design_variables_dimension=self.n_design_variables_dimension,
                n_generation=self.n_ga_generation,
                n_population=self.n_ga_population))
            pop, _ = opt.run()
            x = np.array([list(ind) for ind in pop])
            y = np.array([ind.fitness.values for ind in pop])
            #kmeans = KMeans(n_clusters=self.n_new_samples, n_jobs=-1)
            kmeans = MiniBatchKMeans(n_clusters=8, batch_size=self.batch_size)
            kmeans.fit(x, y)
            new_samples = kmeans.cluster_centers_
            
        return torch.from_numpy(new_samples.astype(np.float32))

 
    def optimize(self):
        self._initialize()
        for bayesian_optimization_iter in range(
                self.bayesian_optimization_iter_max):

            self._train_likelihood()

            print('bayesian opt Iter %d/%d' % (
                bayesian_optimization_iter + 1,
                self.bayesian_optimization_iter_max))

            # if self._judge_termination():
            #     break

            self.new_x = self._find_new_sample()
            self.train_x = torch.cat((self.train_x, self.new_x), dim=0)
            self.train_y = \
                torch.cat((self.train_y,
                           torch.from_numpy(self.evaluation_function(self.new_x).T)), dim=0)
        return self.train_x, self.train_y


'''
experiment
'''
starttime = datetime.datetime.now()


if __name__ == "__main__":

    # multi objective genetic algorithm (NSGA2) is implemented with 'DEAP'
    # Gaussian Process model is implemented with 'gpytorch'
    opt = MultiObjectiveBayesianOpt(
        evaluation_function=zdt1,
        surrogate_model=ExactGPModel,
        optimizer=NSGA2,
        acquisition=acquisition.ei,
        n_objective_dimension=2,
        n_design_variables_dimension=30,
        n_initial_sample=128,#64,128
        n_new_samples=50,
        bayesian_optimization_iter_max=1,
        likelihood_optimization_iter_max=5000,#500,5000,10000
        likelihood_optimization_criteria=1e-6,
        n_ga_population=16,
        n_ga_generation=50,#100
        batch_size=10
    )
    result2 = opt.optimize()

    front2 = np.array(result2[1])
    
    plt.scatter(front2[:, 0], front2[:, 1], c="b")
    plt.axis("tight")
    #print(result)
    #plt.savefig('fig.png')
    #np.savetxt('result_x.csv', result[0].numpy(), delimiter=',')
    #np.savetxt('result_y.csv', result[1].numpy(), delimiter=',')
    plt.show()
    
endtime = datetime.datetime.now()
print ((endtime - starttime).seconds)

