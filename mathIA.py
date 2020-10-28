import matplotlib.pyplot as plt
import numpy as np
import random
import csv
random.seed(1234563)

#calculates the new proportions of proliferators after each timestep
def calc_props(function_coefficients, timesteps):
    pro_props_array = []

    for i in range(timesteps):
        pro_prop = function_coefficients[0]*(pow(i,2)) + function_coefficients[1]*i + function_coefficients[2]
        pro_props_array.append(pro_prop)

    return pro_props_array

#calculates the fitness score of one generation
def calc_fitness(given_data, function_coefficients, timesteps):
    pro_props_array= calc_props(function_coefficients, timesteps)
    residual_sum = 0
    abs_residual_sum = 0

    for i in range(len(given_data)):
        residual_sum += ((pro_props_array[i] - given_data[i]) ** 2)
        abs_residual_sum += abs(pro_props_array[i] - given_data[i])

    #This is the RSS (residual sum of squares). We want to minimze this value
    #fitness = round(residual_sum, 10)

    #This is the MSE (we want to minimize this value)
    fitness = round(((1/timesteps) * residual_sum), 10)

    #This is the LAD (least absolute deviation)
    #fitness = round(abs_residual_sum, 10)

    return fitness

#runs the program for a number of generations
def run_generations(given_data, num_parallel_gen, timesteps):
    data = []
    for i in range(num_parallel_gen):
        function_coefficients = [round(random.uniform(-.00001,.00001), 10),round(random.uniform(-.00001,.00001), 10),round(random.uniform(-.00001,.00001), 10)]
        fitness = calc_fitness(given_data, function_coefficients, timesteps)
        data.append([function_coefficients, fitness])
    return data

def mutate(row, given_data):
    ''' Mutates input row '''

    function_coefficients, _ = row

    def mutate_coefficient(coeff, lower=-.00005, upper=.00005):
        ''' Takes a coefficient and mutates it slightly within some bound '''

        new_coeff = round(coeff + random.uniform(lower, upper), 10)
        return new_coeff

    mutation_strategy = random.randint(1,4)

    if (mutation_strategy in (1,2,3,4)):
        function_coefficients = [mutate_coefficient(function_coefficients[0]), mutate_coefficient(function_coefficients[1]),mutate_coefficient(function_coefficients[2])]

    mutated_row = [function_coefficients, calc_fitness(given_data, function_coefficients, 258)]
    return mutated_row

#Mutations mechanism. Will add crossover
def evolve(data, given_data):
    #print("Data: ", data)
    sorted_data = sorted(data, key=lambda x: x[1])
    mutated_data = sorted_data[:]

    num_to_keep = 10

    for i in range(num_to_keep):
        mutated_data[num_to_keep+i] = mutate(mutated_data[i], given_data)
    return mutated_data

def main():
    f = open('Pro_Freq_IA.csv', encoding='utf-8-sig')
    given_data = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
    for row in given_data:
        given_data = list(row)

    # given_data = [-1 * y for y in given_data]

    timesteps = len(given_data) # Should be 258
    assert timesteps == 258, "Timesteps are not equal to 258"

    # This is an initial set of 20 fitness scores for 20 parallel generations
    initial_data = run_generations(given_data, 20 , timesteps)

    # Now we will mutate this data and keep the generations with the smallest (best) fitness scores
    mutated_data = initial_data[:]

    num_generations = 2000
    for i in range(num_generations):
        mutated_data = evolve(mutated_data, given_data)
            
    initial_data = sorted(initial_data, key=lambda x: x[1])
    mutated_data = sorted(mutated_data, key=lambda x: x[1])

    # Print out initial data and final data
    print("\n############# Initial Data #############\n")
    for i in initial_data:
        print(i)

    print("\n############# Mutated data after {} generations #############\n".format(num_generations))
    for i in mutated_data:
        print(i)

    legend = []

    for i in range(0, len(mutated_data)-10):
        data_to_plot = mutated_data[i]
        pro_props_array = calc_props(data_to_plot[0], timesteps)
        print("data: ", pro_props_array)
        return
        legend.append(mutated_data[i][-1])
        plt.plot(pro_props_array)

    plt.title('Proportion of Proliferators vs Time: MSE Fitness Function')
    plt.legend(legend, loc="upper left")
    plt.ylabel("Proportion of Proliferators")
    plt.xlabel("Timesteps")
    plt.show()

main()
