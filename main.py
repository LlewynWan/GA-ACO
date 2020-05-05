import sys
import numpy as np
from function import randfunc

from animation import animator

from algorithm import GeneticAlgorithm as GA
from algorithm import AntColonyOptimisation as ACO


def AcoAnimate(length=128):
    x = np.linspace(0,1,length)
    y = np.linspace(0,1,length)
    X,Y = np.meshgrid(x,y)
    Z = randfunc(X, Y)

    aco = ACO(Z,length=length)
    aco_seq = aco.simulate()

    sequence = []
    for seq in aco_seq:
        x = np.asarray([ant[0] for ant in seq])
        y = np.asarray([ant[1] for ant in seq])
        sequence.append([x,y])

    AcoAnimator = animator(X,Y,Z,sequence,fig=1,name="ACO")
    AcoAnimator.render()


def GaAnimate(bitlen=7):
    x = np.linspace(0,1,1<<bitlen)
    y = np.linspace(0,1,1<<bitlen)
    X,Y = np.meshgrid(x,y)
    Z = randfunc(X, Y)

    ga = GA(Z,bitlen=bitlen)
    ga_seq = ga.evolve()

    sequence = []
    for seq in ga_seq:
        mask = (1 << bitlen) - 1
        x = seq & mask
        mask <<= bitlen
        y = (seq & mask) >> bitlen
        sequence.append([x,y])

    GaAnimator = animator(X,Y,Z,sequence,fig=2,name="GA",interval=10)
    GaAnimator.render()


def main():
    if len(sys.argv) < 2:
        print("Algorithm Not Specified")
    else:
        spec = sys.argv[1]
        if spec == "ACO":
            AcoAnimate()
        elif spec == "GA":
            GaAnimate()
        else:
            print("Algorithm Not Explicitly Specified")

if __name__ == '__main__':
    main()
