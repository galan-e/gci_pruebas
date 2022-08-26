#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 12:16:39 2022

@author: egalan
"""

import time
import resource
import multiprocessing
import numpy as np
from math import sin, cos, sqrt
from niapy.task import OptimizationType, Task
from niapy.problems import Problem
from niapy.algorithms.basic import (
    GreyWolfOptimizer,
    GeneticAlgorithm,
    ParticleSwarmOptimization,
    HarrisHawksOptimization,
)

# ************************************
N_points = 1500  # Numero de puntos aleatorios
population_size = 20
max_iters = 20
procesadores = 32
# ************************************


rv = 9


def MTH(theta, d, a, alph):  # Matriz de transformacion homogenea
    if alph == 90:
        landa = 0  # cos(alph)
        miu = +1  # sin(alph)
    elif alph == -90:
        landa = 0  # cos(alph)
        miu = -1  # sin(alph)
    elif alph == 0:
        landa = +1  # cos(alph)
        miu = 0  # sin(alph)

    cos_theta = cos(theta)
    sin_theta = sin(theta)
    Qi = np.matrix(
        [
            [cos_theta, -landa * sin_theta, miu * sin_theta, a * cos_theta],
            [sin_theta, landa * cos_theta, -miu * cos_theta, a * sin_theta],
            [0, miu, landa, d],
            [0, 0, 0, 1],
        ]
    )
    return np.matrix(Qi.round(rv))


def beta_AB(q_min, q_max):
    # Calculate the alpha and beta values for beta distribution
    # Cao2011,pg: 6
    # http://dx.doi.org/10.5772/45686
    q_range = q_max - q_min
    if q_range > 0 and q_range < pi / 2:
        a, b = 0.4, 0.4
    elif q_range >= pi / 2 and q_range < pi:
        u = (q_range / (5 * pi)) + 0.3
        a, b = u, u
    elif q_range >= pi and q_range < 3 * pi / 2:
        u = (3 * q_range / (5 * pi)) - 0.1
        a, b = u, u
    elif q_range >= (3 * pi / 2) and q_range <= 2 * pi:
        u = (2 * q_range / (5 * pi)) + 0.2
        a, b = u, u

    return (a, b)


def q_beta(q_min, q_max, N):
    if q_min == 0 and q_max == 0:
        q = np.zeros([N, 1])
    else:
        a, b = beta_AB(q_min, q_max)
        q = q_min + (q_max - q_min) * np.random.beta(a, b, N)
        q = np.reshape(q, [N, 1])
    return q


pi = 3.141592653589793
q1_min = -90 * (pi / 180)
q1_max = 90 * (pi / 180)

q2_min = -60 * (pi / 180)
q2_max = 120 * (pi / 180)

q3_min = -60 * (pi / 180)
q3_max = 120 * (pi / 180)

q4_min = -180 * (pi / 180)
q4_max = 180 * (pi / 180)

q5_min = -90 * (pi / 180)
q5_max = 90 * (pi / 180)

q6_min = -180 * (pi / 180)
q6_max = 180 * (pi / 180)

q_ranges = np.matrix(
    [
        [q1_min, q1_max],
        [q2_min, q2_max],
        [q3_min, q3_max],
        [q4_min, q4_max],
        [q5_min, q5_max],
        [q6_min, q6_max],
    ]
)

# G6R.q_beta(self, q_min, q_max, N)


P_dist = "beta"

if P_dist == "beta":

    q6_beta = q_beta(q_ranges[5, 0], q_ranges[5, 1], N_points)
    q5_beta = q_beta(q_ranges[4, 0], q_ranges[4, 1], N_points)
    q4_beta = q_beta(q_ranges[3, 0], q_ranges[3, 1], N_points)
    q3_beta = q_beta(q_ranges[2, 0], q_ranges[2, 1], N_points)
    q2_beta = q_beta(q_ranges[1, 0], q_ranges[1, 1], N_points)
    q1_beta = q_beta(q_ranges[0, 0], q_ranges[0, 1], N_points)
elif P_dist == "line":

    q6_beta = np.linspace(q_ranges[5, 0], q_ranges[5, 1], N_points)
    q5_beta = np.linspace(q_ranges[4, 0], q_ranges[4, 1], N_points)
    q4_beta = np.linspace(q_ranges[3, 0], q_ranges[3, 1], N_points)
    q3_beta = np.linspace(q_ranges[2, 0], q_ranges[2, 1], N_points)
    q2_beta = np.linspace(q_ranges[1, 0], q_ranges[1, 1], N_points)
    q1_beta = np.linspace(q_ranges[0, 0], q_ranges[0, 1], N_points)

# Saving q vectors for validation

q_save = np.matrix(
    [
        q1_beta.flatten(),
        q2_beta.flatten(),
        q3_beta.flatten(),
        q4_beta.flatten(),
        q5_beta.flatten(),
        q6_beta.flatten(),
    ]
).T

np.savetxt(
    "q_beta_values.cvs", q_save, delimiter=",",
)

N = (1 / 6) * np.identity(6)  # Matriz (1/n)*Inxn


time_start = time.perf_counter()  # Timer Start

#%%
class G6RGCI(Problem):
    def __init__(self, dimension, lower=0.10, upper=1.0, *args, **kwargs):
        super().__init__(dimension, lower, upper, *args, **kwargs)
        self.Links = [0, 0, 0, 0, 0, 0]
        self.d1 = 0
        self.a2 = 0
        self.d4 = 0
        self.d6 = 0

    def KCI(self, i):
        q = [
            q1_beta[i],
            q2_beta[i],
            q3_beta[i],
            q4_beta[i],
            q5_beta[i],
            q6_beta[i],
        ]
        # Standar DH
        A1 = MTH(q[0], self.d1, 0, 90)
        A2 = MTH(q[1], 0, self.a2, 0)
        A3 = MTH(q[2], 0, 0, 90)
        A4 = MTH(q[3], self.d4, 0, -90)
        A5 = MTH(q[4], 0, 0, 90)
        A6 = MTH(q[5], self.d6, 0, 0)
        #  Geometric Jacobian Calculation --Siciliano page 112
        T01 = np.matrix(A1)
        T02 = np.matrix(A1 * A2)
        T03 = np.matrix(A1 * A2 * A3)
        T04 = np.matrix(A1 * A2 * A3 * A4)
        T05 = np.matrix(A1 * A2 * A3 * A4 * A5)
        T06 = np.matrix(A1 * A2 * A3 * A4 * A5 * A6)

        z0 = np.matrix([[0], [0], [1]])
        P0 = np.matrix([[0], [0], [0]])

        P1 = T01[0:3, 3].round(rv)
        R1 = T01[0:3, 0:3].round(rv)
        z1 = R1 * z0

        P2 = T02[0:3, 3].round(rv)
        R2 = T02[0:3, 0:3].round(rv)
        z2 = R2 * z0

        P3 = T03[0:3, 3].round(rv)
        R3 = T03[0:3, 0:3].round(rv)
        z3 = R3 * z0

        P4 = T04[0:3, 3].round(rv)
        R4 = T04[0:3, 0:3].round(rv)
        z4 = R4 * z0

        P5 = T05[0:3, 3].round(rv)
        R5 = T05[0:3, 0:3].round(rv)
        z5 = R5 * z0

        P6 = T06[0:3, 3].round(rv)

        j6 = np.vstack((np.cross(z5.T, (P6 - P5).T).T, z5)).round(rv)
        j5 = np.vstack((np.cross(z4.T, (P6 - P4).T).T, z4)).round(rv)
        j4 = np.vstack((np.cross(z3.T, (P6 - P3).T).T, z3)).round(rv)
        j3 = np.vstack((np.cross(z2.T, (P6 - P2).T).T, z2)).round(rv)
        j2 = np.vstack((np.cross(z1.T, (P6 - P1).T).T, z1)).round(rv)
        j1 = np.vstack((np.cross(z0.T, (P6 - P0).T).T, z0)).round(rv)

        J0 = np.matrix(np.hstack((j1, j2, j3, j4, j5, j6)))  #%Base-frame OK
        # print("J0 : \n", J0)
        # Analytical Jacobian
        # print("R6: \n", R6)

        J = J0
        V = J * N * J.T
        VV = np.linalg.inv(J) * N * np.linalg.inv(J.T)
        K = (1 / 6) * sqrt(np.trace(V) * np.trace(VV))

        return 1 / K

    def _evaluate(self, x):
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]

        g1 = -((x1 / x2) - 1.1)
        g2 = (x1 / x2) - 2.0
        g3 = -((x2 / x3) - 1.1)
        g4 = (x2 / x3) - 2.0
        pp = 10 ^ 7
        g = [g1, g2, g3, g4]
        penalty = [0, 0, 0, 0]
        for i in range(4):
            if g[i] > 0:
                penalty[i] = pp * g[i]

        self.d1 = 30.0
        self.a2 = x1 * 100
        self.d4 = x2 * 100
        self.d6 = x3 * 100
        self.Links = [self.d1, self.a2, 0, self.d4, 0, self.d6]

        pool = multiprocessing.Pool(procesadores)
        K = pool.map(self.KCI, range(N_points))
        GCI = np.mean(K) - sum(penalty)
        pool.close()
        # print("GCI", GCI)
        # print("K size ", len(K))
        return GCI


def GWO_GCI_Optimiazation():
    G6R_GCI_opt = G6RGCI(dimension=3)

    time_start = time.perf_counter()  # Timer Start

    print("GWO start\n")
    task_gwo = Task(
        problem=G6R_GCI_opt,
        max_iters=max_iters,
        optimization_type=OptimizationType.MAXIMIZATION,
    )
    algo_gwo = GreyWolfOptimizer(population_size)
    best_gwo = algo_gwo.run(task_gwo)
    best2_gwo = [str(best_gwo)]
    print("Best solution  GWO", best_gwo)
    np.savetxt("GWO_GCI_best.txt", best2_gwo, fmt="%s")
    np.savetxt(
        "GWO_GCI_convergence_data_values.cvs",
        task_gwo.convergence_data(),
        delimiter=",",
    )
    print("GWO done\n")
    time_elapsed = time.perf_counter() - time_start
    memMb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0
    print("GWO %5.5f secs %5.5f MByte \n" % (time_elapsed, memMb))
    time_memory = [time_elapsed, memMb]
    np.savetxt("GWO_GCI_time.txt", time_memory, fmt="%s")


def GA_GCI_Optimization():
    G6R_GCI_opt = G6RGCI(dimension=3)
    time_start = time.perf_counter()  # Timer Start

    print("GA start\n")
    task_ga = Task(
        problem=G6R_GCI_opt,
        max_iters=max_iters,
        optimization_type=OptimizationType.MAXIMIZATION,
    )
    algo_ga = GeneticAlgorithm(population_size)
    best_ga = algo_ga.run(task_ga)
    best2_ga = [str(best_ga)]
    print("Best solution  GA", best_ga)
    np.savetxt("GA_GCI_best.txt", best2_ga, fmt="%s")
    np.savetxt(
        "GA_GCI_convergence_data_values.cvs", task_ga.convergence_data(), delimiter=","
    )
    print("GA done\n")

    time_elapsed = time.perf_counter() - time_start
    memMb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0
    print("GA %5.5f secs %5.5f MByte \n" % (time_elapsed, memMb))
    time_memory = [time_elapsed, memMb]
    np.savetxt("GA_GCI_time.txt", time_memory, fmt="%s")


def PSO_GCI_Optimization():
    G6R_GCI_opt = G6RGCI(dimension=3)
    time_start = time.perf_counter()  # Timer Start

    print("PSO start\n")
    task_pso = Task(
        problem=G6R_GCI_opt,
        max_iters=max_iters,
        optimization_type=OptimizationType.MAXIMIZATION,
    )
    algo_pso = ParticleSwarmOptimization(population_size)
    best_pso = algo_pso.run(task_pso)
    best2_pso = [str(best_pso)]
    print("Best solution  PSO", best_pso)
    np.savetxt("PSO_GCI_best.txt", best2_pso, fmt="%s")
    np.savetxt(
        "PSO_GCI_convergence_data_values.cvs",
        task_pso.convergence_data(),
        delimiter=",",
    )
    print("PSO done\n")
    time_elapsed = time.perf_counter() - time_start
    memMb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0
    print("PSO %5.5f secs %5.5f MByte \n" % (time_elapsed, memMb))
    time_memory = [time_elapsed, memMb]
    np.savetxt("PSO_GCI_time.txt", time_memory, fmt="%s")


def HHO_GCI_Optimization():
    G6R_GCI_opt = G6RGCI(dimension=3)
    time_start = time.perf_counter()  # Timer Start

    print("HHO start\n")
    task_hho = Task(
        problem=G6R_GCI_opt,
        max_iters=max_iters,
        optimization_type=OptimizationType.MAXIMIZATION,
    )
    algo_hho = HarrisHawksOptimization(population_size)
    best_hho = algo_hho.run(task_hho)
    best2_hho = [str(best_hho)]
    print("Best solution  HHO", best_hho)
    np.savetxt("HHO_GCI_best.txt", best2_hho, fmt="%s")
    np.savetxt(
        "HHO_GCI_convergence_data_values.cvs",
        task_hho.convergence_data(),
        delimiter=",",
    )
    print("HHO done\n")

    time_elapsed = time.perf_counter() - time_start
    memMb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0
    print("HHO %5.5f secs %5.5f MByte \n" % (time_elapsed, memMb))
    time_memory = [time_elapsed, memMb]
    np.savetxt("HHO_GCI_time.txt", time_memory, fmt="%s")


time_startt = time.perf_counter()  # Timer Start

GWO_GCI_Optimiazation()
time.sleep(10)
GA_GCI_Optimization()
time.sleep(10)
PSO_GCI_Optimization()
time.sleep(10)
HHO_GCI_Optimization()
time.sleep(10)


time_elapsed = time.perf_counter() - time_startt
memMb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0
print("COMPLETE PROG %5.5f secs %5.5f MByte" % (time_elapsed, memMb))
time_memory = [time_elapsed, memMb]
np.savetxt("TOTAL_time.txt", time_memory, fmt="%s")
time_memory = [time_elapsed, memMb]
np.savetxt("TOTAL_time.txt", time_memory, fmt="%s")
memMb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0
print("COMPLETE PROG %5.5f secs %5.5f MByte" % (time_elapsed, memMb))
time_memory = [time_elapsed, memMb]
np.savetxt("TOTAL_time.txt", time_memory, fmt="%s")
time_memory = [time_elapsed, memMb]
np.savetxt("TOTAL_time.txt", time_memory, fmt="%s")
