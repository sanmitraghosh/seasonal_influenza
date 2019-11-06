import icuh
import numpy as np
import matplotlib.pyplot as plt
import time


x = np.array([0.53, 0.30, 0.00001, 0.4, 0.0008, 25,  0.07, 15])
def test_add():
    icu_sim = np.array(icuh.ICUHsim(x))
    plt.plot(icu_sim)
    plt.show()
    print(icu_sim)
    print(icu_sim.shape)


if __name__ == '__main__':

    start = time.time()
    test_add()
    end = time.time()
    print(end - start)
