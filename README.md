## About The Project
This is a repository implementing the ASQP-RL

### Setting up the datasets
* Flights - the data can be downloaded from https://github.com/IDEBench/IDEBench-public/tree/master/data and queries from datasets/flights2/generated_queries
* IMDB-JOB - the data & queries can be downloaded from https://github.com/gregrahn/join-order-benchmark
* MAS - the data can be taken from http://academic.research.microsoft.com/ and the queries can be taken from /datasets/mas

### Preparing the Table & Queries:
* First create your joined table
* make sure to add a numeric PK column starting from 0
* Create a .sql file containing queries over that table
* Use `queries_from_file_executor.py` method `start()` to turn the sql file into .pkl format
* For different train-test partitions permute the queries in your .sql file - ASQP-RL takes the first queries as the test set
* You will see a new folder under checkpoints that contains the .pkl files - pass that as an argument when you want to run PPO over these queries

### Running ASQP-RL:
* First configure config.json
  * under `dbConfig` put the db connection details
  * under `queryConfig` put table details (pivot is the numeric PK col)
  * under `samplerConfig` put the viewSize (F in the paper) and the `testSize` (how many queries should be reserved for test)
* Run `ray_sampler.py` with arguments:
  * --k K: The size of the subset to choose (memory size in the paper)
  * --alg ALG: The algorithm to use, can be one of: [A3C, PPO, APEX_DQN]
  * --env ENV: The environment to use, can be one of: [CHOOSE_K, DROP_ONE] (where choose_k is named GSL in the paper)
  * --steps STEPS: How many episodes to run
  * --checkpoint CHECKPOINT: Which folder under /checkpoints the queries should be taken from - results will also be saved here (under a subfolder) 
  * --test: Run in test mode, use this if you want to test an existing model. By default, we both train and test
  * --trial_name <NAME>: Use this if using "--test", set to the subfolder under /checkpoints/<CHECKPOINT> that was created when you trained the model 

