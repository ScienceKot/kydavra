'''
Created with love by Sigmoid

@Author - Păpăluță Vasile - vpapaluta06@gmail.com
'''
# Importing all needed libraries
import random
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
class GeneticAlgorithmSelector:
    def __init__(self, nb_children : int = 4, nb_generation : int = 200, scoring_metric = accuracy_score, max : bool = True) -> None:
        '''
            Setting up the model
        :param nb_children: integer, default = 4
            The number of children created at every generation
        :param nb_generation: integer, default = 200
            The number of generations created to find the best feature combination
        :param scoring_metric: function, default = sklearn.metrics.accuracy_score
            The sklearn like scoring metric
        :param max: boolean, default = True
            Used to define whatever we need the biggest of the smallest values computed by self.scoring_metric
        '''
        self.nb_children = nb_children
        self.nb_generation = nb_generation
        self.scoring_metric = scoring_metric
        self.max = max
    def inverse(self, value : int) -> int :
        '''
            This function inverses the value coming to function
        :param value: integer, o or 1
            The value that should be inverted
        :return: integer
            Return the inverse of :param value (if value = 0 ==> 1, else 0)
        '''
        if value == 0:
            return 1
        else:
            return 0
    def cross_over(self, population : list) -> list:
        '''
            This function apply the crossing-over process on 2 arrys-like lists
        :param population: 2-d array like list
            The population with 2 array that will be uses to create new individuals in the population
        :return: 2-d array like list
            Return the population with parents and their children after corssing-over
        '''
        # Creating an empty list for the new generation
        new_generation = []

        # Adding nb_children children to the new generatin
        for i in range(self.nb_children//2):
            first = random.randrange(0, len(self.X_columns)-1)
            second = random.randrange(0, len(self.X_columns)-1)
            if first > second:
                first, second = second, first
            new_generation.append(population[0][0:first] + population[1][first: second] + population[0][second:])
            new_generation.append(population[1][0:first] + population[0][first: second] + population[1][second:])

        # Adding the new generation to the population
        for gene in new_generation:
            population.append(gene)
        return population
    def mutate(self, gene : list) -> list:
        '''
            This function generates a random mutation on a gene
        :param gene: 1-d array like list
            The list with zeros and ones that will be mutated
        :return: 1-drray like list
            The gene list after mutation
        '''
        # Choosing a random place to generate a mutation
        mutation_locus = random.randrange(0, len(self.X_columns)-1)

        # Generating a mutation
        gene[mutation_locus] = self.inverse(gene[mutation_locus])
        return gene
    def gene_to_cols(self, gene : list) -> list:
        '''
            This function convert the zeros and ones list to columns list
        :param gene: 1-d array like list
            The list with zeros and ones that must be transformed in a columns sequence
        :return: 1-d array like list
            The lust with columns that will go to the model
        '''
        cols = []
        for i in range(len(gene)):
            if gene[i] == 1:
                cols.append(self.X_columns[i])
        return cols
    def select(self, algo, dataframe : 'pd.DataFrame', y_column : str, test_size=0.2):
        '''
            This function selects the best columns
        :param algo: sklearn algorithm class
            An sklearn algorithm class
        :param dataframe: pandas DataFrame
            Data Frame on which the algorithm is applied
        :param y_column: string
            The column name of the value that we what to predict
        :param test_size: float, default = 0.2
            The percent of data set that will be used to find the trained algorithm accuracy
        :return: list
            The list of columns selected by algorithm
        '''
        # Getting the list with names of columns without the target one
        self.X_columns = [col for col in dataframe.columns if col != y_column]

        # Generating the empty population and a temporal one
        population = []
        temp = []
        for i in range(len(self.X_columns)):
            temp.append(random.choice([0, 1]))
        population.append(temp)
        temp = [self.inverse(x) for x in population[-1]]
        population.append(temp)

        # Creating new generations and testing in searching for individual with the best metric
        for gen in range(self.nb_generation):
            # Creating an empty list with the metrics
            acc = []

            # Generating the first cross-over
            population = self.cross_over(population)

            # Generating some mutations
            for i in range(2, len(population)):
                population[i] = self.mutate(population[i])

            # Measuring the accuracy of every individual in the population
            for gene in population:
                if set(gene) == set([0]):
                    continue
                X = dataframe[self.gene_to_cols(gene)].values
                y = dataframe[y_column].values
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
                algo.fit(X_train, y_train)
                y_pred = algo.predict(X_test)
                acc.append(self.scoring_metric(y_test, y_pred))

            # Selecting the 2 best members of the population based on whatever the metric's property
            if self.max:
                res = sorted(range(len(acc)), key=lambda sub: acc[sub])[-2:]
            else:
                res = sorted(range(len(acc)), key=lambda sub: acc[sub])[:2]

            # Creating a new population with the 2 best members
            population = [population[res[0]], population[res[1]]]
        return self.gene_to_cols(population[0])