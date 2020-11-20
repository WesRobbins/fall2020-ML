from MLP import *
import os

class GeneticAlgorithm:

    def __init__(self, dataclass, class_type):
        self.initialize_parameters()
        self.dataclass = dataclass
        self.class_type = class_type
        self.build_population()
        self.run_networks()


    def initialize_parameters(self):
        self.pop_size = 20

    def build_population(self):
        self.population = []
        for i in range(self.pop_size):
            mlp = MLP(self.dataclass, self.class_type)
            self.population.append(mlp)

    def run_networks(self):
        fitness_list = []
        print(self.population)
        for network in self.population:
            fitness_list.append(network.test(self.dataclass.df))
        self.population = [network for _, network in sorted(zip(fitness_list, self.population))]
        print(fitness_list)
        print(self.population)


