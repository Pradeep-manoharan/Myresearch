




import numpy as np



if __name__ == "__main__":
    all_to_all = np.ones((10,10))
    print(all_to_all)
    print(np.identity(10))
    print(all_to_all-np.identity(10))
    print(-10 * (all_to_all-np.identity(10)))