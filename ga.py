import matplotlib.pyplot as plt
import numpy as np
import random
random.seed(1234563)

#calculates the new proportions of strategies after each timestep
def calc_props(init_prop_pro, init_prop_eng, payoff_matrix, timesteps, alpha):
    print ("payoff matrices: ", payoff_matrix)
    pro_props_array = []
    eng_props_array = []
    
    pro_prop = init_prop_pro
    eng_prop = init_prop_eng

    pro_props_array.append(pro_prop)
    eng_props_array.append(eng_prop)

    for i in range(timesteps):
        pro_expected_value = (pro_prop * payoff_matrix[0][0]) + (eng_prop * payoff_matrix[0][1])
        eng_expected_value = (pro_prop * payoff_matrix[1][0]) + (eng_prop * payoff_matrix[1][1])

        # avg = init_prop_pro * pro_expected_value + init_prop_eng * eng_expected_value
        # change_in_pro = init_prop_pro * (pro_expected_value-avg/avg)
        # change_in_eng = init_prop_eng * (eng_expected_value-avg/avg)

        # pro_ab = init_prop_pro + change_in_pro
        # eng_ab = init_prop_eng + change_in_eng

        pro_ab = alpha * (pro_expected_value/(pro_expected_value + eng_expected_value))
        eng_ab = alpha * (eng_expected_value/(pro_expected_value + eng_expected_value))

        if pro_expected_value >= eng_expected_value:
            pro_prop += pro_ab
            eng_prop -= eng_ab
        else:
            pro_prop -= pro_ab
            eng_prop += eng_ab

        pro_props_array.append(pro_prop)
        eng_props_array.append(eng_prop)

    return pro_props_array, eng_props_array

#calculates the fitness score of one generation
def calc_fitness(init_prop_pro, init_prop_eng, payoff_matrix, timesteps, alpha):
    pro_props_array, eng_props_array = calc_props(init_prop_pro, init_prop_eng, payoff_matrix, timesteps, alpha)
    #data = pd.read_excel(/Users/aashiagrawal/Documents/IMO_projects/AMB_ga/Sample_Data.xlsx)

    #average of Jack's data (hardcoded for now)
    pro_avg_actual = .75630
    eng_avg_actual = .24370
    #average of the data from calc_props
    pro_avg_experimental = sum(pro_props_array)/len(pro_props_array)
    eng_avg_experimental = sum(eng_props_array)/len(eng_props_array)
    #difference between my data and Jack's data
    pro_difference = abs(pro_avg_actual - pro_avg_experimental)
    eng_difference = abs(eng_avg_actual - eng_avg_experimental)
    #the difference between my data and Jack's data for the proliferators and engineers will be the same value since the data is symmetric
    #Since the difference between the data is the same, it doesn't matter which strategy's mean difference I use to determine the fitness value
        
    fitness = round( 100 * (1 - pro_difference), 5) #could also use eng_difference but it doesn't matter. Subtracted from 1 because the greatest difference possible is 1
    return fitness

#runs the program for a number of generations
def run_generations(num_parallel_gen, timesteps):
    data = []
    init_prop_pro_array = []
    init_prop_eng_array = []
    payoff_matrix_array = []
    for i in range(num_parallel_gen):
        #random initial proportions
        init_prop_pro = round(random.random(), 5)
        init_prop_eng = round(1- init_prop_pro, 5)
        #payoff matrix with random numbers btwn 0-1
        payoff_matrix = [[round(random.random(), 5),round(random.random(), 5)],[round(random.random(), 5),round(random.random(), 5)]]
        fitness = calc_fitness(init_prop_pro, init_prop_eng, payoff_matrix, timesteps, .001)
        data.append([init_prop_pro, init_prop_eng, payoff_matrix, fitness])
    return data

def crossover():
    ''' Performs crossover'''

    pass

def mutate(row):
    ''' Mutates input row '''
    pro_prop, eng_prop, payoff_matrix, _ = row
    def mutate_prop(prop, lower=-0.3, upper=0.3):
        ''' Takes a probability and mutates it slightly within some bound '''
        if ((1 - prop) < upper):
            upper = 1 - prop
        if (prop < upper):
            lower = -1 * prop
        new_prop = round(prop + random.uniform(lower, upper), 5)
        return new_prop
    
    # 25% probability of modifying only proportions (either add or subtract)
    # 25% prob of modifying payoff matrix
    # 50% prob of mutating both

    mutation_strategy = random.randint(1,4)
    if (mutation_strategy in (1,3,4)):
        pro_prop = mutate_prop(pro_prop)
        eng_prop = round(1 - pro_prop, 5)
    if (mutation_strategy in (2,3,4)):
        payoff_matrix = [[mutate_prop(payoff_matrix[0][0]), mutate_prop(payoff_matrix[0][1])],[mutate_prop(payoff_matrix[1][0]), mutate_prop(payoff_matrix[1][1])]]
    
    mutated_row = [pro_prop, eng_prop, payoff_matrix, calc_fitness(pro_prop, eng_prop, payoff_matrix, 2478, .0005)]
    return mutated_row

#Mutations mechanism. Will add crossover
def evolve(data):
    sorted_data = sorted(data, key=lambda x: x[3])
    mutated_data = sorted_data[:]

    for i in range(len(sorted_data) - 10):
        mutated_data[i] = mutate(sorted_data[i])
    return mutated_data

def main():
    initial_data = run_generations(20, 2478)
    mutated_data = initial_data

    num_generations = 200
    for i in range(num_generations):
        mutated_data = evolve(mutated_data)

    initial_data = sorted(initial_data, key=lambda x: x[3])
    mutated_data = sorted(mutated_data, key=lambda x: x[3])

    # Print out initial data and final data
    print("\n############# Initial Data #############\n")
    for i in initial_data:
        print(i)

    print("\n############# Mutated data after {} generations #############\n".format(num_generations))
    for i in mutated_data:
        print(i)

    legend = []
    for i in range(len(mutated_data)-10, len(mutated_data)):
        data_to_plot = mutated_data[i]
        pro_props_array, eng_props_array = calc_props(data_to_plot[0], data_to_plot[1], data_to_plot[2], 2478, 0.0005)
        print("Mean of mutated data :", np.mean(pro_props_array))
        legend.append(mutated_data[i][-1])
        plt.plot(pro_props_array)


    plt.legend(legend, loc="top left")
    
    plt.show()

main()
