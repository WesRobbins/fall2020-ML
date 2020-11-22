from MLP import *
import random
import time

class GeneticAlgorithm:

    def __init__(self, dataclass, class_type):
        self.initialize_parameters()
        self.dataclass = dataclass
        self.df = dataclass.df
        self.class_type = class_type
        self.build_population(dataclass)
        self.run()



    def initialize_parameters(self):
        self.pop_size = 20
        self.cutoff = .4
        self.mutation_rate = .2
        self.mutation_amount = .2
        self.generations = 40
        self.crossover_rate = .8

    def build_population(self, dataclass):
        """Creates the initial population of neural networks"""
        self.population = [MLP(dataclass, self.class_type) for i in range(self.pop_size)]

    def run(self):
        for i in range(self.generations):
            start_time = time.time()
            avg_fitness = self.fitness()
            end_time = time.time()
            print(f"Time for fitness = {end_time - start_time}")
            print(f"Average fitness for generation {i+1}: {avg_fitness}")
            self.selection()
            self.crossover()
            self.mutation()

    def fitness(self):
        """Runs the dataframe through each neural network in the population, which assigns each
        neural network a fitness value. Then returns average fitness for the population"""

        return sum([nn.test(self.df) for nn in self.population]) / self.pop_size

    def selection(self):
        """Uses tournament selection to select (mostly) the strongest individuals from
        the population"""

        selected_pop = []
        while len(selected_pop) < self.pop_size:
            candidate1, candidate2 = random.choice(self.population), random.choice(self.population)
            if candidate1.fitness < candidate2.fitness:
                selected_pop.append(candidate1)
            else:
                selected_pop.append(candidate2)

        self.population = selected_pop

    def crossover(self):
        """Creates the new population of children by combining weights of two parents to
        form a child"""

        new_population = []
        #Creates n_pop_size new children
        for _ in range(self.pop_size // 2):
            child = []
            child2 = []
            #Picks a random father and mother from the selected population
            father_weights = random.choice(self.population).get_weights()
            mother_weights = random.choice(self.population).get_weights()
            #Verifies father and mother are not the same
            while mother_weights == father_weights:
                mother_weights = random.choice(self.population).get_weights()
            #Creates a new child using uniform crossover from father and mother weights
            if random.random() < self.crossover_rate:
                for j in range(len(father_weights)):
                    if random.getrandbits(1):
                        child.append(father_weights[j])
                        child2.append(mother_weights[j])
                    else:
                        child.append(mother_weights[j])
                        child2.append(father_weights[j])
                new_population.extend([child, child2])
            else:
                new_population.append(father_weights)
                new_population.append(mother_weights)

        self.population = new_population

    def mutation(self):
        """Goes through the individuals in the new population and randomly changes
        certain weights by some random value"""

        new_pop = []
        #print(self.population)
        for individual in self.population:

            for i in range(len(individual)):
                if random.random() < self.mutation_rate:
                    individual[i] += random.uniform(-self.mutation_amount, self.mutation_amount)
                    #print(individual)
            x = MLP(self.dataclass, self.class_type)
            x.set_weights(individual)
            new_pop.append(x)
        self.population = new_pop



