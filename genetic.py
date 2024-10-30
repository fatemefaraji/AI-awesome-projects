from random import randint as rnd, shuffle
import matplotlib.pyplot as plt
import time

N = 4  # board size
PS = 6  # population size
MR = 0.2  # mutation rate should be between 0 and 1
EPOCH = 200  # number of generations

# initialize population
def init_population(n, ps):
    population_list = []
    for i in range(ps):
        member = [rnd(0, n - 1) for _ in range(n)]
        population_list.append(member + [None])  # None is added to store fitness
    return population_list

def cross_over(population_list, n, ps):
    for i in range(0, ps, 2):
        # Split genes between two parents
        child1 = population_list[i][:n // 2] + population_list[i + 1][n // 2:n] + [None]
        child2 = population_list[i + 1][:n // 2] + population_list[i][n // 2:n] + [None]
        population_list.append(child1)
        population_list.append(child2)
    return population_list

def mutation(population_list, n, ps, mr):
    choosen_ones = list(range(ps, ps * 2))  # select offspring
    shuffle(choosen_ones)
    choosen_ones = choosen_ones[:int(ps * mr)]  # select members for mutation

    for i in choosen_ones:
        cell = rnd(0, n - 1)  # randomly select a position in the member
        val = rnd(0, n - 1)  # new random value for mutation
        population_list[i][cell] = val  # apply mutation to the selected position
    return population_list

# calculate the number of conflicts between Qs
def fitness(population_list, n):
    length = len(population_list)
    for i in range(length):
        conflict = 0
        for j in range(n):
            for k in range(j + 1, n):
                # column
                if population_list[i][j] == population_list[i][k]:
                    conflict += 1
                # diagonal
                if abs(j - k) == abs(population_list[i][j] - population_list[i][k]):
                    conflict += 1
        population_list[i][-1] = conflict  # save the conflict count (fitness)
    return population_list

# representation of final solution
def show(solution, n):
    plt.figure(figsize=(5, 5))
    for i in range(n + 1):
        plt.plot([0, n * 2], [i * 2, i * 2], color='black')  # Horizontal lines
        plt.plot([i * 2, i * 2], [0, n * 2], color='black')  # Vertical lines
    for i in range(n):
        plt.scatter([i * 2 + 1], [solution[i] * 2 + 1], color='red', s=200)  # Place the queens
    plt.show()

start_time = time.time()

# Main
current_population = init_population(N, PS)
current_population = fitness(current_population, N)
current_population = sorted(current_population, key=lambda x: x[-1])

# check if solution is found in the initial population
if current_population[0][-1] == 0:
    print("solution found at the initial population stage: ", current_population[0])
else:
    for i in range(EPOCH):
        current_population = cross_over(current_population, N, PS)
        current_population = mutation(current_population, N, PS, MR)
        current_population = fitness(current_population, N)
        current_population = sorted(current_population, key=lambda x: x[-1])

        print(f"generation {i + 1} - best cost (conflicts): {current_population[0][-1]}")

        if current_population[0][-1] == 0:
            print(f"solution found at generation {i + 1}: {current_population[0]}")
            show(current_population[0], N)
            break
        else:
            print(f"best solution so far: {current_population[0]}")
    else:
        print("sorry, we couldn't find a solution!")


# calculating time
end_time = time.time()
espended_time = end_time - start_time
print(f"espended time: {espended_time:.4f} seconds")

# test part for  random mutation selection
x = list(range(PS, PS * 2))
shuffle(x)
x = x[:int(PS * MR)]
print(x)
