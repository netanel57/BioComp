import copy
from abc import ABC

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import abc
# from tqdm import tqdm
import tqdm


class GeneticAlgorithmProblem(metaclass=abc.ABCMeta):
    def __init__(self, seed=None, min_max='max', min_value=-np.inf, max_value=np.inf):
        self.seed = seed
        self.min_max = min_max
        # np.random.seed(self.seed)
        self.random = np.random.RandomState(self.seed)
        self.random_state = self.random.get_state()
        # we might not need min and max value
        self.min_value = min_value
        self.max_value = max_value
        #

    @abc.abstractmethod
    def fitness(self):
        raise NotImplementedError('Please implement a fitness function.')

    @abc.abstractmethod
    def crossover(self, other):
        raise NotImplementedError('Please implement a crossover function.')

    @abc.abstractmethod
    def mutation(self, mutation_rate=0.05):
        raise NotImplementedError('Please implement a mutation function.')


class MagicSquareProblem(GeneticAlgorithmProblem):
    def __init__(self, size, seed=None):
        # we might not need min and max value
        super().__init__(seed=seed, min_max='min', min_value=0, max_value=(size**3) * (size**2 + 1) / 2)
        self.size = size
        self.sub_constant = size**2 + 1
        self.sub_square_constant = 2 * self.sub_constant
        self.constant = size * self.sub_constant / 2
        self.square = self.random.permutation(range(1, size*size + 1)).reshape((size, size))

    def fitness(self):
        # get sums of columns and rows
        cols_abs = np.abs(self.square.sum(axis=0) - self.constant).sum()
        rows_abs = np.abs(self.square.sum(axis=1) - self.constant).sum()
        # get major diagonals
        diag1 = np.diag(self.square)
        diag2 = np.diag(np.rot90(self.square, 1))
        # get sums of major diagonals
        diag1_abs = abs(diag1.sum() - self.constant)
        diag2_abs = abs(diag2.sum() - self.constant)

        if self.size % 2 == 1:
            return sum([cols_abs, rows_abs, diag1_abs, diag2_abs])
        else:
            # gets pairs on major diagonals
            pairs1 = sum([abs(diag1[i::self.size//2].sum() - self.sub_constant) for i in range(self.size//2)])
            pairs2 = sum([abs(diag2[i::self.size//2].sum() - self.sub_constant) for i in range(self.size//2)])

            # get sum of every 2x2 sub-square with wraparound
            sub_squares_sum = sum(self._get_wrapped_2x2_subsquares())

            # returns the sum of all the differences to their target; perfect square is fitness = 0
            return sum([sub_squares_sum, diag1_abs, diag2_abs, pairs1, pairs2, cols_abs, rows_abs])

    def crossover(self, other, crossover_points=1):
        # cross over we defined as in the lecture with multiple crossover points,
        # while fixing only the values from the other parent to make it valid
        assert self.square.shape == other.square.shape, "need to be same shape."
        self.random.set_state(self.random_state)
        flatten_square = self.square.flatten()
        other_flatten_square = other.square.flatten()
        # generating crossover points
        points = np.sort(self.random.choice(range(1, self.square.size), size=crossover_points, replace=False))
        points = np.concatenate(([0], points, [self.square.size]))
        # choosing whether to use the prime parent first or second
        choice = self.random.choice(['first', 'second'])
        result_flatten_square = np.empty_like(flatten_square)
        place_parent_1 = np.empty_like(flatten_square)
        # loop on the crossover points and use the correct parent
        for i in range(len(points) - 1):
            start, end = points[i], points[i + 1]
            if (i % 2 == 0 and choice == 'first') or (i % 2 == 1 and choice == 'second'):
                result_flatten_square[start:end] = flatten_square[start:end]
                place_parent_1[start:end] = True
            elif (i % 2 == 1 and choice == 'first') or (i % 2 == 0 and choice == 'second'):
                result_flatten_square[start:end] = other_flatten_square[start:end]
                place_parent_1[start:end] = False

        unique_elements = set(np.arange(1, self.square.size + 1))
        seen = set()
        duplicates = list()
        dup_dict = dict()
        # look for duplicates from the other parent
        for i, val in enumerate(result_flatten_square):
            if val not in seen:
                if not place_parent_1[i]:
                    dup_dict[val] = i
                seen.add(val)
            elif place_parent_1[i]:
                duplicates.append(dup_dict[val])
            else:
                duplicates.append(i)
        # get all missing values shuffle them and input them in the duplicate locations
        missing = list(unique_elements - seen)
        self.random.shuffle(missing)
        result_flatten_square[duplicates] = missing
        self.square = result_flatten_square.reshape((self.size, self.size))
        self.random_state = self.random.get_state()

        return self

    def mutation(self, mutation_rate=0.05):
        # mutation we defined here as swapping two places
        self.random.set_state(self.random_state)
        # randomize numbers for each cell
        rand_n = self.random.random((self.size, self.size))
        # find all cells to be mutated
        mutators = np.where(rand_n.flatten() < mutation_rate)[0]

        # if there is an odd number add the next number in line or remove if at max size
        if mutators.size % 2 == 1:
            mutators_l = mutators.tolist()
            if mutators.size != self.square.size:
                next = rand_n.flatten()[np.where(rand_n.flatten() >= mutation_rate)[0]].argmin()
                mutators_l.append(next)
            elif mutators.size == self.square.size:
                next = rand_n.flatten()[np.where(rand_n.flatten() < mutation_rate)[0]].argmax()
                mutators_l.remove(next)
            mutators = np.array(mutators_l)

        # shuffle indexes to swap every pair
        self.random.shuffle(mutators)

        # swap every pair
        flatten_square = self.square.flatten()
        for i in range(0, mutators.size, 2):
            tmp_value = flatten_square[mutators[i]]
            flatten_square[mutators[i]] = flatten_square[mutators[i+1]]
            flatten_square[mutators[i+1]] = tmp_value
        self.square = flatten_square.reshape((self.size, self.size))
        self.random_state = self.random.get_state()
        return self

    def _get_wrapped_2x2_subsquares(self):
        # helper function to get sub-squares
        subsquares = []
        for i in range(self.size):
            for j in range(self.size):
                sub_square = self.square[np.ix_([i, (i + 1) % self.size], [j, (j + 1) % self.size])]
                sub_square_sum = sub_square.sum()
                sub_abs = abs(sub_square_sum - self.sub_square_constant)
                subsquares.append(sub_abs)
        return subsquares

    # maybe define solver/learner here or create new class for it

    def __radd__(self, other):
        if isinstance(other, MagicSquareProblem):
            return self.fitness() + other.fitness()
        else:
            return self.fitness() + other

    def __add__(self, other):
        if isinstance(other, MagicSquareProblem):
            return self.fitness() + other.fitness()
        else:
            return self.fitness() + other

    def __eq__(self, other):
        # maybe need to add this depending if we're looking for equivalence solely on fitness
        # np.array_equal(self.square, other.square)
        if isinstance(other, MagicSquareProblem):
            return self.fitness() == other.fitness()
        else:
            return self.fitness() == other

    def __ge__(self, other):
        if isinstance(other, MagicSquareProblem):
            return self.fitness() >= other.fitness()
        else:
            return self.fitness() >= other

    def __le__(self, other):
        if isinstance(other, MagicSquareProblem):
            return self.fitness() <= other.fitness()
        else:
            return self.fitness() <= other

    def __gt__(self, other):
        if isinstance(other, MagicSquareProblem):
            return self.fitness() > other.fitness()
        else:
            return self.fitness() > other

    def __lt__(self, other):
        if isinstance(other, MagicSquareProblem):
            return self.fitness() < other.fitness()
        else:
            return self.fitness() < other

    def __ne__(self, other):
        if isinstance(other, MagicSquareProblem):
            return self.fitness() != other.fitness()
        else:
            return self.fitness() != other

    def __str__(self):
        return self.square.__str__()

    def __repr__(self):
        return self.__str__()


def selection_min(population, population_fitness, randomer):
    max_f = max(population_fitness)
    reverse_f = [(max_f - f) + 1 for f in population_fitness]
    reverse_total_f = sum(reverse_f)
    norm_total_f = [f/reverse_total_f for f in reverse_f]
    sel1, sel2 = randomer.choice(population, size=2, replace=False, p=norm_total_f)
    return sel1, sel2


class GeneticAlgorithm:
    def __init__(self, problem, problem_args=None, crossover_points=1, mutation_rate=0.05, elitism=0, learning_type=None, selection_method=selection_min,
                 pop_size=100, population_split=4, population_seeds=None, seed=None):
        self.pop_size = pop_size
        self.problem = problem
        self.min_max = problem(**problem_args).min_max
        self.elitism = elitism
        self.selection_method = selection_method
        self.population_seeds = population_seeds
        if not self.population_seeds is None:
            self.population = [self.problem(**problem_args, seed=s) for s in self.population_seeds]
        else:
            self.population = [self.problem(**problem_args, seed=None) for s in range(self.pop_size)]
        # self.sorted_population = list()
        # self.sorted_population_fitness = list()
        self.learning_type = learning_type
        self.crossover_points = crossover_points
        self.mutation_rate = mutation_rate
        self.running_mutation_rate = mutation_rate
        self.temp_change = 10
        self.running_temp_change = 10
        self.random = np.random.RandomState(seed)
        self.random_state = self.random.get_state()
        self.population_split = population_split

    def generation_step(self, population):
        new_population = list()
        sorted_population = sorted(population, reverse=self.min_max == 'max')
        sorted_population_fitness = [p.fitness() for p in sorted_population]
        if self.elitism != 0:
            for i in range(self.elitism):
                new_population.append(sorted_population[i])
        for i in range(len(sorted_population) - self.elitism):
            new_population.append(self.offspring_creation(sorted_population, sorted_population_fitness))
        return new_population

    def offspring_creation(self, sorted_population, sorted_population_fitness):
        self.random.set_state(self.random_state)
        p1, p2 = self.selection_method(sorted_population, sorted_population_fitness, self.random)
        self.random_state = self.random.get_state()
        p1_copy = copy.deepcopy(p1)
        offspring = p1_copy.crossover(p2, crossover_points=self.crossover_points)
        offspring.mutation(mutation_rate=self.running_mutation_rate)
        return offspring

    def learning_step(self):
        if self.learning_type:
            # TODO: create darwinian and lamarkian learning steps
            if self.learning_type == 'darwinian':
                pass
            elif self.learning_type == 'lamarkian':
                pass
        # else:
        #     self.sorted_population = sorted(self.population, reverse=self.min_max == 'max')
        #     self.sorted_population_fitness = [p.fitness() for p in self.sorted_population]

    # def generation_step(self):
    #     new_population = list()
    #     if self.elitism != 0:
    #         for i in range(self.elitism):
    #             new_population.append(self.sorted_population[i])
    #     for i in range(self.pop_size - self.elitism):
    #         new_population.append(self.offspring_creation(self.sorted_population, self.sorted_population_fitness))
    #     self.population = new_population
    #
    # def offspring_creation(self, sorted_population, sorted_population_fitness):
    #     self.random.set_state(self.random_state)
    #     p1, p2 = self.selection_method(sorted_population, sorted_population_fitness, self.random)
    #     self.random_state = self.random.get_state()
    #     p1_copy = copy.deepcopy(p1)
    #     offspring = p1_copy.crossover(p2, crossover_points=self.crossover_points)
    #     offspring.mutation(mutation_rate=self.running_mutation_rate)
    #     return offspring
    #
    # def learning_step(self):
    #     if self.learning_type:
    #         # TODO: create darwinian and lamarkian learning steps
    #         if self.learning_type == 'darwinian':
    #             pass
    #         elif self.learning_type == 'lamarkian':
    #             pass
    #     else:
    #         self.sorted_population = sorted(self.population, reverse=self.min_max == 'max')
    #         self.sorted_population_fitness = [p.fitness() for p in self.sorted_population]

    def play(self, max_steps=100):
        # TODO: create a procedure that deals with premature convergence
        min_f = min(self.population)
        t = tqdm.trange(max_steps, desc="Result = ")
        # for i in tqdm(range(max_steps)):
        for i in t:
            self.learning_step()
            if i % 10 == 0 and i != 0:
                self.population = self.generation_step(self.population)
                # print('migration')
            else:
                for split in range(1, self.population_split+1):
                    start = (split - 1) * self.pop_size // self.population_split
                    end = split * self.pop_size // self.population_split
                    self.population[start:end] = self.generation_step(self.population[start:end])
            curr = min(self.population)
            curr_max = max(self.population)
            curr_average = sum(self.population) / self.pop_size
            t.set_description(f'Best = {curr.fitness()}, Worst = {curr_max.fitness()}, Avg = {curr_average}')
            # if min_f > curr:
            #     min_f = curr
            #     # last_gen_improvement = i
            #     # self.running_mutation_rate = self.mutation_rate
            #     # self.running_temp_change = self.temp_change
            #     print(min_f.fitness(), i, curr_max.fitness())
            if curr == 0:
                # print('stopped at:', i)
                break
            # if i - last_gen_improvement + 1 == self.running_temp_change:
            #     self.running_mutation_rate += self.mutation_rate
            #     self.running_temp_change += self.temp_change
            #     print(i, last_gen_improvement, self.running_mutation_rate)
            #     last_gen_improvement = i

        return sorted(self.population, reverse=self.min_max == 'max')[0], sorted(self.population, reverse=self.min_max == 'max')[0].fitness()


# couple of tests will delete later
ga = GeneticAlgorithm(MagicSquareProblem, problem_args={'size': 5}, elitism=2, crossover_points=1,
                      mutation_rate=0.05, population_seeds=np.arange(42, 142), pop_size=100, seed=32)
print(ga.play(max_steps=500))
# a = [1,2,3,4,5,6,7,8,9,10,11,12]
# a_len = len(a)
# for i in range(1,4+1):
#     a[(i-1)*a_len//4:i*a_len//4] = [77*i] * (a_len//4)
# print(a)
# 6, 500 gens, mr = 0.01, cop = 1, e = 2, f = 176, expanding mr
# 6, 500 gens, mr = 0.01, cop = 1, e = 2, f = 130

# msp2 = MagicSquareProblem(4, 52)
# msp = MagicSquareProblem(4, 42)
# print(sum([msp,1]))
# print(msp.mutation())
# print(msp.mutation())
# print(msp.mutation())
# print(msp.mutation())