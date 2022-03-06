a solution is presented as a directory

the directory must contain one [yaml](https://en.wikipedia.org/wiki/YAML) file:
`conf.yaml`

and in the file, on the top level, 4 keys can have values: `setup-env-command`, `etl-command`, `process-command`
and `cleanup-command`. 

- `image` the docker image base where the solution can run
- `setup-command` sets up the environment where the other commands can run.
- `etl-command` runs, when the data is already accessible by the solution, in this case in a `hotel_table.csv` in the root of the solution. the etl command can do whatever it wants with the data to prepare it for the process command
- when `process-command` runs, an additional `inputs.json` file is also present in the solution root. your task is to make this command write out the answers to the queries found in inputs into an `outputs.json` file in the root of the solution, as fast as possible. this is the only mandatory value
- `cleanup-command` runs after everything is done


solutions will be evaluated based on:
- base speed
- scaling with size of query
- scaling with size of data
