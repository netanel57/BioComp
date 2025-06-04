import copy
import time
from abc import ABC
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import abc
# from tqdm import tqdm
import tqdm
import itertools

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
    def __init__(self, size, seed=None, square=None, mode="standard"):
        # we might not need min and max value
        super().__init__(seed=seed, min_max='min', min_value=0, max_value=(size**3) * (size**2 + 1) / 2)
        self.size = size
        self.mode = mode
        self.sub_constant = size**2 + 1
        self.sub_square_constant = 2 * self.sub_constant
        self.constant = size * self.sub_constant / 2
        if square is not None:
            self.square = square
        else:
            self.square = self.random.permutation(range(1, size * size + 1)).reshape((size, size))
        self.computed_fitness = None

    def fitness(self):
        if self.computed_fitness is not None:
            return self.computed_fitness
        # get sums of columns and rows
        cols_abs = np.abs(self.square.sum(axis=0) - self.constant).sum()
        rows_abs = np.abs(self.square.sum(axis=1) - self.constant).sum()
        # get major diagonals
        diag1 = np.diag(self.square)
        diag2 = np.diag(np.rot90(self.square, 1))
        # get sums of major diagonals
        diag1_abs = abs(diag1.sum() - self.constant)
        diag2_abs = abs(diag2.sum() - self.constant)

        if self.mode == "standard":
            f = sum([cols_abs, rows_abs, diag1_abs, diag2_abs])
            self.computed_fitness = f
            return f
        elif self.mode == "most_perfect":
            # gets pairs on major diagonals
            pairs1 = sum([abs(diag1[i::self.size//2].sum() - self.sub_constant) for i in range(self.size//2)])
            pairs2 = sum([abs(diag2[i::self.size//2].sum() - self.sub_constant) for i in range(self.size//2)])

            # get sum of every 2x2 sub-square with wraparound
            sub_squares_sum = np.abs(self._get_wrapped_2x2_subsquares().sum(axis=(2, 3)) - self.sub_square_constant).sum()

            # returns the sum of all the differences to their target; perfect square is fitness = 0
            f = sum([sub_squares_sum, diag1_abs, diag2_abs, pairs1, pairs2, cols_abs, rows_abs])
            self.computed_fitness = f
            return f

    def crossover(self, other, crossover_points=1):
        # cross over we defined as in the lecture with multiple crossover points,
        # while fixing only the values from the other parent to make it valid
        assert self.square.shape == other.square.shape, "need to be same shape."
        self.computed_fitness = None
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
        self.computed_fitness = None
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
        # Get all top-left corners of 2x2 blocks
        row_idx = np.arange(self.size).reshape(self.size, 1)
        col_idx = np.arange(self.size).reshape(1, self.size)

        # Compute 2x2 block indices with wraparound
        r0 = row_idx
        r1 = (row_idx + 1) % self.size
        c0 = col_idx
        c1 = (col_idx + 1) % self.size

        # Use advanced indexing to gather all 2x2 blocks
        top_left = self.square[r0, c0]
        top_right = self.square[r0, c1]
        bottom_left = self.square[r1, c0]
        bottom_right = self.square[r1, c1]

        # Stack into shape (self.size, W, 2, 2)
        blocks = np.stack([
            np.stack([top_left, top_right], axis=-1),
            np.stack([bottom_left, bottom_right], axis=-1)
        ], axis=-2)

        return blocks

    def optimization_action(self, steps=1, learning='lamarkian'):
        # OK SO LET'S CHOOSE RANDOM K INSTEAD OF ALL THE PAIRS
        best_score = self.fitness()
        best_square = self.square.copy()
        true_old_square = self.square.copy()
        K = self.size * self.size
        indices = list(itertools.product(range(self.size), repeat=2))
        rng = self.random
        for _ in range(steps):
            changed = False #changed=improved
            chosen = rng.choice(len(indices), size=K, replace=False)
            for idx in chosen:
                i1, j1 = indices[idx]
                i2, j2 = rng.randint(0, self.size), rng.randint(0, self.size)
                while i2 == i1 and j2 == j1:
                    i2, j2 = rng.randint(0, self.size), rng.randint(0, self.size)

                #SWAPP
                candidate = best_square.copy()
                candidate[i1, j1], candidate[i2, j2] = candidate[i2, j2], candidate[i1, j1]
                self.square = candidate
                self.computed_fitness = None
                new_score = self.fitness()

                if new_score < best_score:
                    best_score = new_score
                    best_square = candidate.copy()
                    changed= True

            if not changed:
                break
        if learning == 'lamarkian':
            self.square = best_square
        elif learning == 'darwinian':
            self.square = true_old_square
        self.computed_fitness = best_score
        return self

    # def optimization_action_2(self, steps=1, learning='lamarkian'):
    #     # TODO: need to optimize this to have most-perfect squares not be so slow
    #     best_score = self.fitness()
    #     best_square = self.square.copy()
    #     true_old_square = self.square.copy()
    #     k = 0
    #     changed = True
    #     while changed and k < steps:
    #         changed = False
    #         k += 1
    #         old_square = best_square.copy()
    #         diff_fit = np.empty_like(old_square)
    #
    #         cols_diff = old_square.sum(axis=0) - self.constant
    #         rows_diff = old_square.sum(axis=1) - self.constant
    #         diag1 = np.diag(old_square)
    #         diag2 = np.diag(np.rot90(old_square, 1))
    #         diag1_diff = diag1.sum() - self.constant
    #         diag2_diff = diag2.sum() - self.constant
    #         for i in range(self.size):
    #             for j in range(self.size):
    #                 # diff = cols_diff[j] + rows_diff[i]
    #                 diff_fit[i, j] = cols_diff[j] + rows_diff[i]
    #                 diffs = [cols_diff[j], rows_diff[i]]
    #                 if i == j:
    #                     diff_fit[i, j] += diag1_diff
    #                     # diff += diag1_diff
    #                     diffs.append(diag1_diff)
    #                     if self.size % 4 == 0:
    #                         pair_diff1 = old_square[i, j] + old_square[
    #                             (i + self.size // 2) % self.size, (j + self.size // 2) % self.size] - self.sub_constant
    #                         diff_fit[i, j] += pair_diff1
    #                         # diff += pair_diff1
    #                         diffs.append(pair_diff1)
    #                 elif i + j == self.size - 1:
    #                     diff_fit[i, j] += diag2_diff
    #                     # diff += diag2_diff
    #                     diffs.append(diag2_diff)
    #                     if self.size % 4 == 0:
    #                         pair_diff2 = old_square[i, j] + old_square[
    #                             (i + self.size // 2) % self.size, (j - self.size // 2) % self.size] - self.sub_constant
    #                         diff_fit[i, j] += pair_diff2
    #                         # diff += pair_diff2
    #                         diffs.append(pair_diff2)
    #                 if self.size % 4 == 0:
    #                     square_diff_1 = old_square[np.ix_([i, (i + 1) % self.size], [j, (j + 1) % self.size])].sum() - self.sub_square_constant
    #                     square_diff_2 = old_square[np.ix_([i, (i - 1) % self.size], [j, (j - 1) % self.size])].sum() - self.sub_square_constant
    #                     square_diff_3 = old_square[np.ix_([i, (i - 1) % self.size], [j, (j + 1) % self.size])].sum() - self.sub_square_constant
    #                     square_diff_4 = old_square[np.ix_([i, (i + 1) % self.size], [j, (j - 1) % self.size])].sum() - self.sub_square_constant
    #                     diff_fit[i, j] += square_diff_1 + square_diff_2 + square_diff_3 + square_diff_4
    #                     # diff += square_diff_1 + square_diff_2 + square_diff_3 + square_diff_4
    #                     diffs.append(square_diff_1)
    #                     diffs.append(square_diff_2)
    #                     diffs.append(square_diff_3)
    #                     diffs.append(square_diff_4)
    #                 # if i == 2 == j:
    #                 #     print(diff_fit[i, j], diff, diff_fit)
    #                 best_option = max(min(old_square[i, j] - np.ceil(np.mean(diffs)), self.size ** 2), 1)
    #                 candidate = old_square.copy()
    #                 point = np.argwhere(old_square == best_option)
    #                 candidate[*point[0]] = old_square[i, j]
    #                 candidate[i, j] = old_square[*point[0]]
    #                 self.square = candidate
    #                 # print(candidate)
    #                 self.computed_fitness = None
    #                 score = self.fitness()
    #                 # diff_p1 = diff_fit[i, j] + len(diffs) * (best_option - candidate[i, j])
    #                 # diff_p2 = diff_fit[*point[0]]
    #                 # score =
    #                 # print(score)
    #                 if score < best_score:
    #                     best_score = score
    #                     changed = True
    #                     best_square = candidate.copy()
    #                 # if abs(diff) > abs(best_diff):
    #                 #     best_diff = diff
    #                 #     point = (i, j)
    #                 #     diff_list = [q for q in diffs if q != 0]
    #
    #         # print(diff_fit)
    #         # print(point)
    #         # print(diff_list)
    #         # print(diff_list[np.abs(diff_list).argmin()])
    #         # next_diff = max(min(old_square[point] - np.ceil(np.mean(diff_list)), self.size ** 2), 1)
    #
    #         # print(next_diff)
    #         # point_2 = np.argwhere(old_square == next_diff)
    #         # print(point_2[0])
    #         # print(old_square[*point_2[0]])
    #         # tmp = old_square[*point_2[0]]
    #         # candidate = old_square.copy()
    #         # candidate[*point_2[0]] = old_square[point]
    #         # candidate[point] = old_square[*point_2[0]]
    #         # self.square = candidate
    #         # score = self.fitness()
    #         # if score < best_score:
    #         #     best_score = score
    #         #     changed = True
    #         #     best_square = candidate.copy()
    #     if learning == 'lamarkian':
    #         self.square = best_square
    #     elif learning == 'darwinian':
    #         self.square = true_old_square
    #     self.computed_fitness = best_score
    #     # self.computed_fitness = None
    #     return self

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
    def __init__(self, problem, problem_args=None, crossover_points=1, mutation_rate=0.05, elitism=0,
                 learning_type=None, learning_cap=None,selection_method=selection_min,
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
        self.learning_type = learning_type
        self.learning_cap = learning_cap
        self.crossover_points = crossover_points
        self.mutation_rate = mutation_rate
        self.running_mutation_rate = mutation_rate
        self.random = np.random.RandomState(seed)
        self.random_state = self.random.get_state()
        self.population_split = population_split

    def generation_step(self, population,fitness_list=None):
        new_population = list()
        sorted_population = sorted(population, reverse=self.min_max == 'max')
        sorted_population_fitness = [p.fitness() for p in sorted_population]
        if fitness_list is not None:
            sorted_idx = sorted(range(len(population)),
                                key=lambda i: fitness_list[i],
                                reverse=(self.min_max == 'max'))
            sorted_population = [population[i] for i in sorted_idx]
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

    def learning_step(self, population):
        if self.learning_type == "lamarkian":
            return [
                p.optimization_action(steps=self.learning_cap, learning=self.learning_type)
                for p in population
            ]

        if self.learning_type == "darwinian":
            self._darwinian_fitness = [
                p.optimization_action(steps=self.learning_cap, learning=self.learning_type).fitness()
                for p in population
            ]
        return population

    def play(self, max_steps=100):
        # TODO: create a procedure that deals with premature convergence
        min_f = min(self.population).fitness()
        t = tqdm.trange(max_steps, desc="Result = ")
        self.running_mutation_rate = self.mutation_rate
        last_gen_improvement = 0
        for i in t:
            self.population = self.learning_step(self.population)
            # ----------------------------------------
            # if i % 10 == 0 and i != 0:
            #     self.population = self.generation_step(self.population)
            #     # print('migration')
            # else:
            #     for split in range(1, self.population_split+1):
            #         start = (split - 1) * self.pop_size // self.population_split
            #         end = split * self.pop_size // self.population_split
            #         self.population[start:end] = self.generation_step(self.population[start:end])
            # -------------------------------------------
            self.population = self.generation_step(self.population)

            curr = min(self.population)
            curr_average = sum(self.population) / self.pop_size
            # print(i - last_gen_improvement)
            if min_f > curr.fitness():
                min_f = curr.fitness()
                last_gen_improvement = i
                self.running_mutation_rate = self.mutation_rate

            elif i - last_gen_improvement >= 25:
                self.running_mutation_rate = self.mutation_rate
                last_gen_improvement = i
            elif i - last_gen_improvement >= 20:
                self.running_mutation_rate = 10 * self.mutation_rate

            t.set_description(f'Best = {curr.fitness()}, Avg = {curr_average}, Mutation rate: {self.running_mutation_rate}')

            if curr == 0:
                break
        if self.learning_type == 'darwinian':
            return (sorted(self.population, reverse=self.min_max == 'max')[0].optimization_action(learning=self.learning_type, steps=self.learning_cap),
                    sorted(self.population, reverse=self.min_max == 'max')[0].fitness())
        return sorted(self.population, reverse=self.min_max == 'max')[0], sorted(self.population, reverse=self.min_max == 'max')[0].fitness()


# couple of tests will delete later
if __name__ == "__main__":
    size = 8
    ga = GeneticAlgorithm(MagicSquareProblem, problem_args={'size': size}, elitism=2, crossover_points=4,
                          mutation_rate=0.05,
                          # learning_type='darwinian',
                          learning_type='lamarkian',
                          learning_cap=1,
                          population_seeds=np.arange(42, 142), pop_size=100, seed=32)
    print(ga.play(max_steps=500))


# a = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
# arr1 = np.array(a).reshape((4,4))
# print(arr1)
# print(arr1.shape)
# for i in range(4):
#     for j in range(4):
#         if i == j:
#             print(arr[np.ix_([i, (i + 1) % 4], [j, (j + 1) % 4])])
#             print(arr[np.ix_([i, (i - 1) % 4], [j, (j - 1) % 4])])
#             print(arr[np.ix_([i, (i - 1) % 4], [j, (j + 1) % 4])])
#             print(arr[np.ix_([i, (i + 1) % 4], [j, (j - 1) % 4])])
# print(arr.argmin())
# print(np.unravel_index(arr.argmax(), arr.shape))
# 0,1,2,3
# 0,1,2,3,4,5,6,7
# def generate_magic_square(size):
#     # square = np.arange(1, size**2+1).reshape((size, size))
#     square = np.array([[ 1,  7,  9, 16],
#      [15, 14,  6,  3],
#      [ 8,  2, 13, 10],
#      [11, 12,  5,  4]])
#     # print(square)
#     for i in range(size//4):
#         temp = square[:, i + size//2].copy()
#         square[:, i + size//2] = square[:, size - i - 1].copy()
#         square[:, size - i - 1] = temp.copy()
#     print(square)
#
#     for i in range(size//4):
#         temp = square[i + size//2, :].copy()
#         square[i + size//2, :] = square[size - i - 1, :].copy()
#         square[size - i - 1, :] = temp.copy()
#     print(square)
#     for i in range(size):
#         for j in range(size):
#             if i % 2 == 0 and j % 2 == 1 and i < size/2:
#                 temp = square[i, j]
#                 square[i, j] = square[(i + size//2) % size, (j + size//2) % size]
#                 square[(i + size//2) % size, (j + size//2) % size] = temp
#             if i % 2 == 1 and j % 2 == 1 and i < size/2:
#                 temp = square[i, j]
#                 square[i, j] = square[(i + size //2) % size, j]
#                 square[(i + size //2) % size, j] = temp
#             if i % 2 == 1 and j % 2 == 0 and j < size/2:
#                 temp = square[i, j]
#                 square[i, j] = square[i, (j + size//2) % size]
#                 square[i, (j + size//2) % size] = temp
#     return square
# indices = list(itertools.product(range(8), repeat=2))
# print(indices)
# print(len(indices))
# print(len([i for i in itertools.combinations(indices, 2)]))
# def extract_2x2_wraparound_blocks(arr):
#     H, W = arr.shape
#     assert H == W, "Matrix must be square"
#
#     # Get all top-left corners of 2x2 blocks
#     row_idx = np.arange(H).reshape(H, 1)
#     col_idx = np.arange(W).reshape(1, W)
#
#     # Compute 2x2 block indices with wraparound
#     r0 = row_idx
#     r1 = (row_idx + 1) % H
#     c0 = col_idx
#     c1 = (col_idx + 1) % W
#
#     # Use advanced indexing to gather all 2x2 blocks
#     top_left     = arr[r0, c0]
#     top_right    = arr[r0, c1]
#     bottom_left  = arr[r1, c0]
#     bottom_right = arr[r1, c1]
#
#     # Stack into shape (H, W, 2, 2)
#     blocks = np.stack([
#         np.stack([top_left, top_right], axis=-1),
#         np.stack([bottom_left, bottom_right], axis=-1)
#     ], axis=-2)
#
#     return blocks  # shape: (H, W, 2, 2)
# print(arr1)
# print(extract_2x2_wraparound_blocks(arr1))
# print(np.abs(extract_2x2_wraparound_blocks(arr1).sum(axis=(2,3)) - 17).sum())
# print(generate_magic_square(4))




