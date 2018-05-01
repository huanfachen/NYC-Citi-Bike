# Citi Bike Sharing Simulation

**Required Modules**

This simulation software requires NumPy version >= 1.7.0.

Note: Georgia Tech's Deepthought cluster has NumPy version 1.4.1, so this software will not run there.

**Running the Simulation**

To execute the simulation, run the following command from the root directory:
`python -m simcode.src.nycbike --loglevel=info`

To execute the simulation with full trace output, run the same command with `loglevel=debug`. Note that execution will take significantly longer because of the large volume of output to the console. The `outputs` directory contains sample log output from a simulation run. 

**Unit Tests**

Software unit tests were written to verify the behavior of each component used in the simulation. The unit tests can be found in the `simcode/tests/` directory.

* `test_engine.py` - Tests the general-purpose discrete event simulation engine.
* `test_nycbike.py` - Tests the individual event handlers in the simulation application.

Individual tests can be executed using the command:  
`python -m [test module]`  
&nbsp; e.g. `python -m simcode.test.test_engine`  
&nbsp;&nbsp;&nbsp;&nbsp; or `python -m simcode.test.test_nycbike`

**Dataset Statistics (Python Notebook)**

To generate statistics from the [NYC Citi bike dataset](http://www.nyc.gov/html/dot/html/bicyclists/bikestats.shtml):

1. Navigate to the `simcode/src/data/trip_statistics/` directory.
2. Unzip the dataset `201801-citibike-tripdata.csv`.
3. In the command line, run the command `jupyter notebook` to start the Python notebook server.
4. In the browser, visit the notebook interface at [localhost:8888](http://localhost:8888) and open `generate_trip_statistics.ipynb`.

**Bike Distribution Optimization (Python Notebook)**

To optimize the initial distribution of bikes used in the simulation:

1. Navigate to the `simcode/src/data/initial_distribution/` directory in the command-line.
2. Run the command `jupyter notebook` to start the Python notebook server. 
4. In the browser, visit the notebook interface at [localhost:8888](http://localhost:8888) and open `optimize_bikes.ipynb`.
