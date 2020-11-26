from MLP import *
import random

class GeneticAlgorithm:
    """A class that trains weights of a neural network using a genetic styled algorithm. Includes
    mutation, selection, and crossover as functions of the algorithm."""

    def __init__(self, dataclass, class_type):
        """Initializes hyperparameters, builds the initial population and begins the generational
        loop of training weights"""
        self.initialize_parameters()
        self.dataclass = dataclass
        self.df = dataclass.df
        self.class_type = class_type
        self.build_population(dataclass)
        self.run()



    def initialize_parameters(self):
        """Initializes various hyperparameters for tuning purposes"""

        self.pop_size = 20
        self.mutation_rate = .1
        self.mutation_amount = .2
        self.generations = 40
        self.crossover_rate = .8

    def build_population(self, dataclass):
        """Creates the initial population of neural networks"""

        self.population = [MLP(dataclass, self.class_type) for i in range(self.pop_size)]

    def run(self):
        """Iterates through the generations, assigning average fitness for a generation and then
        selecting individuals, crossing over genes, and mutating the new population"""

        #Iterates through every generation(hyperparameter)
        for i in range(self.generations):
            avg_fitness = self.fitness()
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
            #Chooses two random candidates
            candidate1, candidate2 = random.choice(self.population), random.choice(self.population)
            #Best candidate goes back into the gene pool
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
                    #Random boolean, either true or false 50% of the time
                    if random.getrandbits(1):
                        child.append(father_weights[j])
                        child2.append(mother_weights[j])
                    else:
                        child.append(mother_weights[j])
                        child2.append(father_weights[j])
                #Put new children in the new population
                new_population.extend([child, child2])
            #Around 20% of the time, do not crossover and just put mother and father back in population
            else:
                new_population.append(father_weights)
                new_population.append(mother_weights)

        self.population = new_population

    def mutation(self):
        """Goes through the individuals in the new population and randomly changes
        certain weights by some random value"""

        #Creates a new population to hold the neural networks
        new_pop = []

        #Iterates over the list of weights in the population
        for individual in self.population:
            #Iterates through each weight
            for i in range(len(individual)):
                #Around %10 percent of the time, mutate a gene by a small amount
                if random.random() < self.mutation_rate:
                    individual[i] += random.uniform(-self.mutation_amount, self.mutation_amount)
            #Create a neural network from the weights
            x = MLP(self.dataclass, self.class_type)
            x.set_weights(individual)
            new_pop.append(x)
        self.population = new_pop



