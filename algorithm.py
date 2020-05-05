import numpy as np
from random import sample, randint, uniform

from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


class GeneticAlgorithm:
    def __init__(self, Z, bitlen=7, pop_size=25, num_iter=200, copy_ratio=0.3, mutate_prob=0.01):
        self.bitlen = bitlen
        self.pop_size = 10
        self.num_iter = 10
        self.copy_ratio = 0.4
        self.mutate_prob = 0.1
        self.Z = Z


    def calc_fitness(self, population):
        mask = (1 << self.bitlen) - 1
        x = population & mask
        mask <<= self.bitlen
        y = (population & mask) >> self.bitlen
        fitness = self.Z[x,y]
        fitness -= np.min(fitness)
        
        return fitness

    
    def crossover(self, parents):
        father, mother = parents
        num_gene = randint(1, self.bitlen//2)
        index = sample(list(range(2*self.bitlen)), num_gene)

        mask = 0
        for ind in index:
            mask |= (1<<ind)
        fgene = father & mask
        mgene = mother & mask
        child1 = (father & ~mask) | mgene
        child2 = (mother & ~mask) | fgene

        return [child1, child2]


    def mutate(self, children):
        new_child = []
        for child in children:
            rand = uniform(0,1)
            if (rand > self.mutate_prob):
                new_child.append(child)
            else:
                num_gene = randint(1, self.bitlen//2)
                index = sample(list(range(2*self.bitlen)), num_gene)

                mask = 0
                for ind in index:
                    mask |= (1<<ind)
                new_child.append(child^mask)
        
        return new_child


    def mate(self, population, probility, num_children):
        children = []
        for i in range(num_children):
            index = np.random.choice(len(population), size=2, p=probility)
            while (index[0] == index[1]):
                index = np.random.choice(len(population), size=2, p=probility)
            parents = population[index]
            children += self.mutate(self.crossover(parents))
        
        return np.asarray(children)


    def evolve(self, return_sequence=True):
        init = sample(list(range(0,1<<(self.bitlen*2))), k=self.pop_size)
        population = np.asarray(init)

        sequence = [population.copy()]
    
        for i in range(self.num_iter):
            fitness = self.calc_fitness(population)
            args = np.argsort(-fitness)
            population = population[args]
            fitness = fitness[args]

            if np.sum(fitness) > 0:
                probs = fitness / (np.sum(fitness))
            else:
                probs = np.ones_like(fitness) * (1/len(fitness))

            num_copy = int(self.pop_size * self.copy_ratio)
            num_child = self.pop_size - num_copy

            population = np.concatenate((population[0:num_copy], self.mate(population, probs, num_child)))
            if return_sequence:
                sequence.append(population.copy())
        
        if return_sequence:
            return sequence
        else:
            return population


class AntColonyOptimisation:
    def __init__(self, Z, length=128, num_ants=15, num_iter=400, alpha=1, beta=1, rou=0.75, P0=0.25, visual=True):
        self.length = length
        self.num_ants = num_ants
        self.num_iter = num_iter
        self.alpha = alpha
        self.beta = beta
        self.rou = rou

        self.Z = Z
        self.P0 = P0
        self.heu = Z - np.min(Z)
        self.tau = np.zeros(num_ants)

        self.ants = []
        for i in range(num_ants):
            x = randint(0,length-1)
            y = randint(0,length-1)
            self.tau[i] += self.heu[x][y]
            self.ants.append([x,y])

    
    def simulate(self, return_sequence=True):
        dx = [-1, 0, 1, 0]
        dy = [0, 1, 0, -1]

        sequence = [self.ants.copy()]

        for i in range(self.num_iter):
            new_ants = []
            probs = (np.max(self.tau)-self.tau) / self.tau
            for ind,ant in enumerate(self.ants):
                x,y = ant
                if probs[ind] > self.P0:
                    while True:
                        d = randint(0,3)
                        nx = x + dx[d]
                        ny = y + dy[d]
                        if nx < 0 or nx >= self.length:
                            continue
                        if ny < 0 or ny >= self.length:
                            continue
                        new_ants.append([nx,ny])
                        break
                else:
                    nx = randint(0, self.length-1)
                    ny = randint(0, self.length-1)
                    new_ants.append([nx,ny])
            for j in range(len(self.ants)):
                ant = self.ants[j]
                new_ant = new_ants[j]
                if self.Z[ant[0]][ant[1]] < self.Z[new_ant[0]][new_ant[1]]:
                    self.ants[j] = new_ants[j]
            for ind,ant in enumerate(self.ants):
                self.tau[ind] = self.tau[ind]*self.rou + self.heu[ant[0]][ant[1]]

            if return_sequence:
                sequence.append(self.ants.copy())

        if return_sequence:
            return sequence
        else:
            return self.ants
