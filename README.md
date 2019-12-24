# Traveling Salesperson Algorithms

This python project implements a variety of traveling salesperson algorithms. It can display the algorithms and download them as mp4s.

## Getting Started

Anaconda would be the optimal way to run. Saving as an mp4 requires ffmpeg. 

## Running from command line

python atlas.py <options>
* -h, --help: show help message
* -a, --algorithm:	name of the traveling salesman algorithm to use. 
	* Iterative algorithms:
		* sim_annealing - simulated annealing - best iterative algorithm
		* aco - ant colony optimization
		* pso - particle swarm  optimization
		* genetic - genetic algorithm with SCX breeding
		* two_opt - 2-opt optimization
		* three_opt - 3-opt optimization
	* Exact algorithms:
		* dynamic - dynamic algorithm - best exact algorithm
		* brute - computes every path and gets best
		* recursive - brute force algorithm based on recursion
		* bnb - branch and bound 
	* Heuristic algorithms:
		* greedy - nearest neighbor algorithm - best heuristic algorithm
		* mst - minimum search tree algorithm
		* christofide - mst algorithm with nodes of odd degree connected
* -s, --show: what to display when computing algorithm
	* best - shows best algorithm after all iterations
	* improve - display any improvements 
	* all - display every iteration of algorithm
	* none - display nothing
* -r, --ret: what to return as output
	* dist: prints distance of shortest path found
	* path: prints order of shortest path found
	* none: prints nothing
* -S, --save: what to save the images displayed as
* -u, --until: how many iterations to do (useless for non iterative algorithms)
* -t, --type: what locations to use for tsp
	* <file_name.csv>: csv file to read location x,y pairs from
	* r: random locations are used
	* c: generate a circle of locations
* -c, --count: how many locations to generate (ignored if reading from a file
* -l, --lo: lower limit for x and y when generating locations (ignored if reading from a file)
* -h, --hi: upper limit for x and y when generating locations (ignored if reading from a file)

## Examples

* python atlas.py --algorithm aco --show improve --save aco --until 1024 --type berlin52.csv
	* ant colony optimization, showing improvements, save as aco.mp4, 1024 iterations, read locations from berlin52.csv
	* ACO was actually able to find the optimal solution in this case. 

![ant colony optimization of berlin fifty two dataset](https://github.com/N8Brooks/algorithms_tsp/blob/master/examples/aco.gif)

* python atlas.py --algorithm sim_annealing --show improve --save sim_annealing --until 100 --type berlin52.csv
	* simulated annealing, showing improvements, save as sim_annealing.mp4, 100 iterations, read locations from berlin52.csv
	* This is a really good algorithm. It is able to find the optimal solution for the berlin52 dataset regularly. 

![simulated annealing on berlin fifty two dataset](https://github.com/N8Brooks/algorithms_tsp/blob/master/examples/sim_annealing.gif)

* python atlas.py --algorithm greedy --save greedy --type r --count 32
	* greedy algorithm, save as greedy.mp4, 32 random locations

![greedy algorithm](https://github.com/N8Brooks/algorithms_tsp/blob/master/examples/greedy.png)

* python atlas.py --algorithm dynamic --save dynamic --type c --count 18
	* dynamic algorithm, save as dynamic.mp4, run on a circle of 18 locations

![dynamic algorithm](https://github.com/N8Brooks/algorithms_tsp/blob/master/examples/dynamic.png)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
