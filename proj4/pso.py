from MLP import *
import random
import time

"""This file Implements Particle Swarm Optimization"""

class PSO:

    def __init__(self, dataclass, class_type):
        print("\n\nParticle Swarm Optimization")
        print("------------------------------------------------")
        self.dataclass = dataclass
        self.class_type = class_type
        self.df = dataclass.df
        self.initialize_parameters()
        self.topology = self.build_topology()

        self.run()


    def initialize_parameters(self):
        self.average_fitness = 1
        self.swarm_size = 10
        self.swarm = self.build_swarm(self.dataclass)
        self.topology_type = "gBEST"
        # hyperparameters
        self.w = .8
        self.c1 = 1.0
        self.c2 = .6

    def build_swarm(self, dataclass):
        """Creates the initial swarm of neural networks"""
        swarm = [MLP(dataclass, self.class_type) for i in range(self.swarm_size)]
        """ randomize weights so not all Nodes are the same to start"""
        for i in swarm:
            weights = i.get_weights()
            for j in range(len(weights)):
                pass
                weights[j] *= random.uniform(-1, 1)
            i.set_weights(weights)
        return swarm

    def build_topology(self):
        if self.topology_type == "lBEST":
            return self.make_lbest()
        elif self.topology_type == "gBEST":
            return self.make_gbest()

    def make_lbest(self):
        """ build ring topology """
        topology = []
        for i in range(self.swarm_size):
            node = Graph_Node(self.swarm[i])
            topology.append(node)
        for i in range(self.swarm_size):
            if i == 0:
                topology[i].neighbours.append(topology[self.swarm_size-1])
                topology[i].neighbours.append(topology[i+1])
            elif i == self.swarm_size-1:
                topology[i].neighbours.append(topology[i-1])
                topology[i].neighbours.append((topology[0]))
            else:
                topology[i].neighbours.append(topology[i-1])
                topology[i].neighbours.append(topology[i+1])

        return topology

    def make_gbest(self):
        topology = []
        for i in range(self.swarm_size):
            node = Graph_Node(self.swarm[i])
            topology.append(node)
        for i in range(self.swarm_size):
            for j in range(self.swarm_size):
                if i != j:
                    topology[i].neighbours.append(topology[j])
        return topology

    def fitness(self):
        """Runs the dataframe through each neural network in the population, which assigns each
        neural network a fitness value. Then returns average fitness for the swarm"""

        return sum([nn.test(self.df) for nn in self.swarm]) / self.swarm_size


    def run(self):
        """ runs until stopping condition is met"""
        t = 0
        while self.average_fitness > .2:
            t += 1
            start_time = time.time()
            self.average_fitness = self.fitness()
            end_time = time.time()
            #print(f"Time for fitness = {end_time - start_time}")
            print(f"Average fitness for time t = {t}: {self.average_fitness}\n")
            """ update personal best"""
            for i in range(self.swarm_size):
                fit = self.topology[i].nn.test(self.df)
                if fit < self.topology[i].pb_fitness:
                    self.topology[i].pb_fitness = fit
                    self.topology[i].pb = self.topology[i].nn.get_weights()
            """ update neighbourhood best"""
            for i in range(self.swarm_size):
                nb_return = self.get_nb(self.topology[i])
                if nb_return[0] < self.topology[i].nb_fitness:
                    self.topology[i].nb_fitness = nb_return[0]
                    self.topology[i].nb = self.topology[i].neighbours[nb_return[1]].pb
            """ update position """
            for i in range(self.swarm_size):
                """ update velocity """
                r1 = random.random()
                r2 = random.random()
                weights = self.topology[i].nn.get_weights()
                for j in range(len(weights)):
                    """ Attribute wise calculation of velocity"""
                    """add inertia term"""
                    new_velocity = self.w*self.topology[i].velocity[j]
                    """add cognitive term"""
                    new_velocity += self.c1*r1*(self.topology[i].pb[j]-weights[j])
                    """add social term"""
                    new_velocity += self.c2*r2*(self.topology[i].nb[j]-weights[j])
                    """ velocity clamping """
                    if new_velocity > 1:
                        new_velocity = 1
                    elif new_velocity < -1:
                        new_velocity = -1

                    self.topology[i].velocity[j] = new_velocity
                """ update position """
                for j in range(len(weights)):
                    weights[j] += self.topology[i].velocity[j]
                self.topology[i].nn.set_weights(weights)



    def get_nb(self, node):
        nb_fit = 9999
        num = ""
        for i in range(len(node.neighbours)):
            #fit = node.neighbours[i].test(self.df)
            fit = node.neighbours[i].pb_fitness
            if fit < nb_fit:
                nb_fit = fit
                num = i
        return nb_fit, num



class Graph_Node:
    def __init__(self, nn):
        self.pb = ""
        self.pb_fitness = 10
        self.nb = ""
        self.nb_fitness = 10
        self.nn = nn
        self.velocity = []
        self.weights = self.nn.get_weights()
        for w in self.weights:
            self.velocity.append(random.uniform(0, .1))
        self.neighbours = []

    def add_neighbour(self, nn):
        self.neighbours.append(nn)