from voronoi_ca import VoronoiCA
import numpy as np

def main():

    # make up data points
    np.random.seed(10)
    
    # run process
    voronoi_ca = VoronoiCA()
    voronoi_ca.proc()
    
    print(1)

if __name__ == "__main__":
	main()