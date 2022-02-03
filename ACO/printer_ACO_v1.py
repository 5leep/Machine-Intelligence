# ANT COLONY OPTIMISATION.

import numpy as np
import pandas as pd
from numpy import inf

q = input("Please input file name with file extension (.csv)")
user1 = pd.read_csv(q)
print("Input to Printer=")
print(user1)
a = pd.read_csv("data_project.csv", skiprows=0, usecols=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36])
c = np.array(a)
# print(a)
b = np.array(a)
# print(b)
data = pd.read_csv("data_project.csv")
z = data['order_num']  # as a Series
y = len(z)

print("Output=", y, "Posters to be printed")
# given values for the problems

d = b

iteration = 100
n_jobs = y
n_colors = 32
# Initialisation

m = n_jobs
n = n_colors
e = .3      # evaporation rate
alpha = 1   # pheromone factor
beta = 3    # visibility factor

# calculating the visibility of the next city visibility(i,j)=1/d(i,j)

visibility = 1 / d
visibility[visibility == inf] = 0

# Initialising pheromone present at the paths to the cities

pheromone = .1 * np.ones((m, n))

# Initialising the route of the ants with size route (n_ants, n_city's + 1)
# + 1 because we want to come back to the source city

route = np.ones((m, n + 1))
for ite in range(iteration):

    route[:, 0] = 1  # initial starting and ending position of every ant '1' i.e city '1'

    for i in range(0):

        temp_visibility = np.array(visibility)  # creating a copy of visibility

        for j in range(n - 1):
            # print(route)

            combine_feature = np.zeros(4)  # Initialising combine_feature array to zero
            cum_prob = np.zeros(4)         # Initialising 'cumulative probability' (cum_prob) array to zeros

            cur_loc = int(route[i, j] - 1)   # current city of the ant

            temp_visibility[:, cur_loc] = 0  # making visibility of the current city as zero

            p_feature = np.power(pheromone[cur_loc, :], beta)  # calculating pheromone feature
            v_feature = np.power(temp_visibility[cur_loc, :], alpha)  # calculating visibility feature

            p_feature = p_feature[:, np.newaxis]  # adding axis to make a size[5,1]
            v_feature = v_feature[:, np.newaxis]  # adding axis to make a size[5,1]

            combine_feature = np.multiply(p_feature, v_feature)  # calculating the combine feature

            total = np.sum(combine_feature)  # sum of all the feature

            prob = combine_feature / total  # finding probability of element prob(i) = combine_feature(i)/total

            cum_prob = np.cumsum(prob)  # calculating cumulative sum
            # print(cum_prob)
            r = np.random.random_sample()  # random no in [0,1)
            # print(r)
            city = np.nonzero(cum_prob > r)[0][0] + 1  # finding the next city having probability higher than random(r)
            # print(city)

            route[i, j] = city  # adding city to route

        left = list(set([i for i in range(0, n + 1)]) - set(route[i, :-5]))[0]  # Finding last non-taken city to route.

        route[i, -5] = left  # adding non-traversed city to route

    rute_opt = np.array(route)  # Initialising optimal route

    dist_cost = np.zeros((m, 1))  # Initialising total_distance_of_tour with zero

    for i in range(m):

        s = 0
        for j in range(n - 1):
            s = s + d[int(rute_opt[i, j]) - 1, int(rute_opt[i, j + 1]) - 1]  # Calculating total tour distance

        dist_cost[i] = s  # storing distance of tour for 'i' ant at location 'i'

    dist_min_loc = np.argmin(dist_cost)  # Finding location of minimum of dist_cost
    dist_min_cost = dist_cost[dist_min_loc]  # Finding min of dist_cost

    best_route = route[dist_min_loc, :]  # Initialising current traversed as best route
    pheromone = (1 - e) * pheromone  # evaporation of pheromone with (1-e)

    for i in range(m):
        for j in range(n - 1):
            dt = 1 / dist_cost[i]
            pheromone[int(rute_opt[i, j]) - 1, int(rute_opt[i, j]) - 1] = pheromone[int(rute_opt[i, j]) - 1, int(
                rute_opt[i, j]) - 1] + dt
            # Updating the pheromone with delta_distance
            # delta_distance will be more with min_dist, i.e adding more weight to that route pheromone

h = (20 * int(dist_min_cost[0]) + 15 * d[int(best_route[-2]) - 1, 0])
print("Number of Violations=", h)
print("Cost of Violations=", 5 * h)

print("Schedule for each colour:")
print("0=", c[0])
print("1=", c[1])
print("2=", c[2])
print("3=", c[3])
print("4=", c[4])
print("5=", c[5])
print("6=", c[6])
print("7=", c[7]) \
 \
    # print('other route with violations and faults :')
# print(rute_opt)
# print()
# print('best path :',best_route)
