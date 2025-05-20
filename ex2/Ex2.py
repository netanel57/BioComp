from abc import ABC

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import abc


class GeneticAlgorithmProblem(metaclass=abc.ABCMeta):
    def __init__(self, seed=None, min_max='max', min_value=-np.inf, max_value=np.inf):
        self.seed = seed
        self.min_max = min_max
        np.random.seed(self.seed)
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
    def __init__(self, size, seed=None, min_max='min', min_value=0):
        # we might not need min and max value
        super().__init__(seed=seed, min_max=min_max, min_value=min_value, max_value=(size**3) * (size**2 + 1) / 2)
        self.size = size
        self.sub_constant = size**2 + 1
        self.sub_square_constant = 2 * self.sub_constant
        self.constant = size * self.sub_constant / 2
        self.square = np.random.permutation(range(1, size*size + 1)).reshape((size, size))

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
        flatten_square = self.square.flatten()
        other_flatten_square = other.square.flatten()
        # generating crossover points
        points = np.sort(np.random.choice(range(1, self.square.size), size=crossover_points, replace=False))
        points = np.concatenate(([0], points, [self.square.size]))
        # choosing whether to use the prime parent first or second
        choice = np.random.choice(['first', 'second'])
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
        np.random.shuffle(missing)
        result_flatten_square[duplicates] = missing
        self.square = result_flatten_square.reshape((self.size, self.size))

        return self.square

    def mutation(self, mutation_rate=0.05):
        # mutation we defined here as swapping two places

        # randomize numbers for each cell
        rand_n = np.random.random((self.size, self.size))
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
        np.random.shuffle(mutators)

        # swap every pair
        flatten_square = self.square.flatten()
        for i in range(0, mutators.size, 2):
            tmp_value = flatten_square[mutators[i]]
            flatten_square[mutators[i]] = flatten_square[mutators[i+1]]
            flatten_square[mutators[i+1]] = tmp_value
        self.square = flatten_square.reshape((self.size, self.size))
        return self.square

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

    def __eq__(self, other):
        # maybe need to add this depending if we're looking for equivalence solely on fitness
        # np.array_equal(self.square, other.square)
        return self.fitness() == other.fitness()

    def __ge__(self, other):
        return self.fitness() >= other.fitness()

    def __le__(self, other):
        return self.fitness() <= other.fitness()

    def __gt__(self, other):
        return self.fitness() > other.fitness()

    def __lt__(self, other):
        return self.fitness() < other.fitness()

    def __ne__(self, other):
        return self.fitness() != other.fitness()

    def __str__(self):
        return self.square.__str__()

    def __repr__(self):
        return self.__str__()


class GeneticAlgorithm:
    def __init__(self, problem, learning_type=None, problem_args=None, pop_size=100):
        self.pop_size = pop_size
        self.problem = problem
        self.min_max = problem.min_max
        # we might not need min and max value
        self.min_value = problem.min_value
        self.max_value = problem.max_value
        #
        self.population = [self.problem(**problem_args) for i in range(pop_size)]
        self.learning_type = learning_type

    def generation_step(self):
        sorted_population = sorted(self.population, reverse=self.min_max == 'max')
        # TODO: max-min range maybe also add helper function for selection
        pass

    def learning_step(self):
        if self.learning_type:
            pass

    def play(self, max_steps=100):
        for i in range(max_steps):
            self.learning_step()
            self.generation_step()


# couple of tests will delete later
msp = MagicSquareProblem(3, 42)
# msp_4 = MagicSquareProblem(3, 42)
# msp_2 = MagicSquareProblem(3, 39)
# msp_3 = MagicSquareProblem(3, 32)
# # for i in range(32):
# tmp = MagicSquareProblem(3, 5)
# print(tmp)
# print(msp)
# # print(msp_2)
# # print(msp_3)
# print(msp.fitness())
# print(tmp.fitness())
# # print(msp_2.fitness())
# # print(msp_3.fitness())
# print(msp == tmp)
# print(msp != tmp)
# print(msp.mutation(mutation_rate=0.03))
# print(msp.mutation(mutation_rate=0.03))
# print(msp.mutation(mutation_rate=0.03))

s1 = MagicSquareProblem(3, 50)
s2 = MagicSquareProblem(3, 32)
print(s1)
print(s2)
print(s1.crossover(s2, 1))
# print(s1.crossover(s2))
# print(s1.crossover(s2))
# print(s1.crossover(s2))
# print(s1.crossover(s2))
# print(s1.crossover(s2))
# print(s1.crossover(s2))
# print(s2.crossover(s1))
# print(s1.crossover(s2))
# print(s1.crossover(s2))
# print(s1.crossover(s2))
# print(s1.crossover(s2))
# print(s1.crossover(s2))
# print(s1.crossover(s2))
# print(s1.crossover(s2))
# print(s1.crossover(s2))

# s2 = np.array([9,8,7,6,5,4,3,2,1])

# number_list = np.arange(1,10)
# res = np.array(s1[:3].tolist() + s2[3:].tolist()).reshape((3,3))
# arg = np.isin(s2[3:], s1[:3])
# arg2 = np.isin([1,2,3,4,5,6,7,8,9], res)
# missing_numbers = number_list[~arg2]
# perm_missing_numbers = np.random.permutation(missing_numbers)
# print(arg)
# print(number_list[~arg2])
# print(res)
# print(perm_missing_numbers)
# print(s2)
# # res2 = np.array(s1[:3].tolist() + s2[3:].tolist()).reshape((3,3))
# s2[3:][arg] = perm_missing_numbers
# fixed_res = np.array(s1[:3].tolist() + s2[3:].tolist()).reshape((3,3))
# print(fixed_res)
# l = [msp, msp_2, msp_3]
# print(sorted(l, reverse=True))
