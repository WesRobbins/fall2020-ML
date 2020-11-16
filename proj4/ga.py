class GeneticAlgorithm:

    def __init__(self):
        self.initialize_parameters()
        self.build_population()

    def initialize_parameters(self):
        self.pop_size = 20

    def build_population(self):
        