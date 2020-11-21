from MLP import *
import random

class GeneticAlgorithm:

    def __init__(self, dataclass, class_type):
        self.initialize_parameters()
        self.dataclass = dataclass
        self.df = dataclass.df
        self.class_type = class_type
        self.build_population(dataclass)
        self.run()



    def initialize_parameters(self):
        self.pop_size = 40
        self.cutoff = .4
        self.mutation_rate = .2
        self.epochs = 20

    def build_population(self, dataclass):
        """Creates the initial population of neural networks"""
        self.population = [MLP(dataclass, self.class_type) for i in range(self.pop_size)]

    def run(self):
        for i in range(self.epochs):
            avg_fitness = self.fitness()
            print(f"Average fitness for epoch {i}: {avg_fitness}")
            self.selection()
            self.crossover()
            self.mutation()

    def fitness(self):
        """Runs the dataframe through each neural network in the population, which assigns each
        neural network a fitness value. Then returns average fitness for the population"""

        return sum([nn.test(self.df) for nn in self.population]) / self.pop_size

    def selection(self):
        """Sorts the population by fitness in ascending order (since we want to minimize loss)
        and selects the first n individuals, determined by our cutoff"""

        self.population.sort(key=lambda x: x.fitness)
        self.population = self.population[:int(self.cutoff * self.pop_size)]
        print(self.population)


    def crossover(self):
        new_population = []
        for _ in range(self.pop_size):
            child = []
            father = random.choice(self.population).get_weights()
            mother = random.choice(self.population).get_weights()
            for j in range(len(father)):
                if random.getrandbits(1):
                    child.append(father[j])
                else:
                    child.append(mother[j])

            new_population.append(child)
        self.population = new_population

    def mutation(self):
        new_pop = []
        for individual in self.population:
            for weight in individual:
                if random.random() < self.mutation_rate:
                    weight += random.uniform(-5, 5)
            x = MLP(self.dataclass, self.class_type)
            x.set_weights(individual)
            new_pop.append(x)
        self.population = new_pop



