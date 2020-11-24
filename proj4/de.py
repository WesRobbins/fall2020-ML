from MLP import *
import random
import time

class DiffEvol:

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
        # beta value in DE equation
        self.beta = 1

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

    def fitness(self):
        """Runs the dataframe through each neural network in the population, which assigns each
        neural network a fitness value. Then returns average fitness for the population"""

        return sum([nn.test(self.df) for nn in self.population]) / self.pop_size

    def selection(self):
        """Uses differences in fitness between candidate vectors to determine whether or not to replace"""

        selected_pop = []
        for i in range(self.pop_size):
            target_vector = self.population[i]

            diff1, diff2, candidate1 = random.choice(self.population), random.choice(self.population), random.choice(self.population)
            while diff1 == diff2 or diff1 == candidate1 or diff2 == candidate1 or candidate1 == target_vector or diff1 == target_vector or diff2 == target_vector:
                # make sure diff1 not equal diff 2 not equal candidate not equal target
                diff1, diff2, candidate1 = random.choice(self.population), random.choice(
                    self.population), random.choice(self.population)

            diff1_weights, diff2_weights, candidate1_weights = diff1.get_weights(), diff2.get_weights(), candidate1.get_weights()
            diff_weights = []
            for i in range(len(diff1_weights)):
                diff_weights.append(diff1_weights[i] - diff2_weights[i])

            trial_weights = []
            for i in range(len(candidate1_weights)):
                trial = candidate1_weights[i] + (self.beta * diff_weights[i])
                trial_weights.append(trial)

            for i in range(len(trial_weights)):
                if random.random() < self.crossover_rate:  # then we'll cross over
                    trial_weights[i] = target_vector.get_weights()[i]

            trial_homie = MLP(self.dataclass, self.class_type)
            trial_homie.set_weights(trial_weights)
            if trial_homie.test(self.df) < target_vector.fitness:
                selected_pop.append(trial_homie)

            else:
                selected_pop.append(target_vector)

        self.population = selected_pop


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


