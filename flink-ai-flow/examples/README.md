# Running Examples

We provide users with some python examples to illustrate the usage of AI Flow and more examples will be added soon.
Before you try the example, please make sure you have installed AI-Flow(locally or in Docker) following the instructions in [QuickStart](https://github.com/alibaba/flink-ai-extended/blob/master/flink-ai-flow/QUICKSTART.md)
Besides, please make sure setting airflow_deploy_path is correct in the project.yaml according to the comments.

### Python Examples

#### [1. Simple Transform Scheduled by AirFlow](python_examples/python_codes/simple_transform_airflow)
This example will read a csv file and calculate the square of each data point and then write to a new file to save results. 
The example codes under `python_examples/python_codes/simple_transform_airflow` dir follows the procedure **read file -> transform -> write file** to achieve the goal.
Besides, it is scheduled by AirFlow.


To run this example:
1. Make sure the AI Flow server, Airflow server and notification server is started on the target machine.

If you are in your local machine, please start them following guidance in [QuickStart](https://github.com/alibaba/flink-ai-extended/blob/master/flink-ai-flow/QUICKSTART.md).

If you are in Docker environment, these services should have been started.

2. Configure and Run script

Configure AirFlow Deploy path in `python_examples/project.yaml`(If using docker, the path has been already set). Then run commands:
```shell
 cd python_examples
 export PYTHONPATH=.
 python python_codes/simple_transform_airflow/simple_transform_airflow.py
```

When the program finishes, you should be able to view transform result in `sink_data.csv` file under `python_examples/python_codes/simple_transform_airflow` dir.

If you do not get the expected output, check logs in [Web UI](127.0.0.1:8080) or in log dir under the AirFlow Deploy dir.

#### [2. Batch Train and Batch Predict Scheduled by AirFlow](python_examples/python_codes/batch_train_batch_predict_airflow)
This example will read a mnist data file and follow the procedure "read data -> train -> evaluate -> validate -> push -> predict" to train an LR model and use it for inference.


To run this example:
1. Make sure the AI Flow server, Airflow server and notification server is started on the target machine.

2. Configure and Run script

Configure AirFlow Deploy path in `python_examples/project.yaml`(If using docker, the path has been already set). Then run commands:
```shell
 cd python_examples
 export PYTHONPATH=.
 python python_codes/batch_train_batch_predict_airflow/sklearn_batch_train_batch_predict.py
```


When the program finishes, from [Web UI](127.0.0.1:8080), you should see it the dag successfully finishes all jobs. 
And you should be able to view the evaluation result of the model in the file `examples/python_examples/python_codes/batch_train_batch_predict_airflow/evaluate_model`, which should look like:

```shell
model version[1] scores: [0.8242636  0.83025462 0.83       0.81981982 0.82314629]
```
Besides, the prediction result is stored in the file `examples/python_examples/python_codes/batch_train_batch_predict_airflow/predict_model`. 
The generated model named by it generated date is saved under dir `examples/python_examples/batch_train_batch_predict_airflow/saved_model`.

If you do not get the expected output, check logs in [Web UI](127.0.0.1:8080) or in log dir under the AirFlow Deploy dir.

#### [3. Stream Train and Stream Predict Scheduled by AirFlow](python_examples/python_codes/stream_train_stream_predict_airflow)



### Simple Examples
See [QuickStart](https://github.com/alibaba/flink-ai-extended/blob/master/flink-ai-flow/QUICKSTART.md).
