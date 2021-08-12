# Pagination recognition

This repository is about my master thesis : 
[Large Scale Web Data API Creation via Automatic Pagination Recognition - A Case Study on Event Extraction](https://docs.google.com/presentation/d/1T-W_U-oDD3wyg3foQwch3hNV-3LntReGQKxvppLS1K8/edit?usp=sharing)

I will introduce the code by following three parts:
1. Pagination Recognition
    * Follow Autopager
    * Use pretrained multilingual LASER
    * Use neural encoding to represent anchor tag(\<a\>) and button tag(\<button\>) 
2. Pagination Recognition Service
    * Use FastAPI to build pagination recognition service 
3. Auto Extraction Flow
    * Use RabbitMQ to trigger auto extraction, and integrat with page-level extraction system: Unsupervised Data ETL system

## Environment preparation
### Package installation
```shell=
pip install -r requirements.txt
```
### Environment activation
```
source {your_ENV}/bin/activate
```
## Pagination Recognition

Baseline model are trained in 'notebooks/Training CRFSuite.ipynb'.

PRNSM are trained in notebooks with prefix 'Custom Training'.

Training/Testing data are located in 'autopager/autopager/data'.

## Pagination Recognition Service

Execute following command under project directory:
```shell=
. run_app.sh
```
Listening port: 9876
You can change any serving method with uvicorn parameters.
### API docs
Open FastAPI docs (localhost:9876/docs) can view request detail.
Request method include:
API router: 'localhost:9876/autopager'
API token: 'fake-API-token'
| Method | path | required auth | description | 
| -------- | -------- | -------- | -- |
| GET     | / | True | Return every prediction records |
| GET     | /{tid} | True | Return {tid} prediction records |
| POST     | / | True | Create a record with URL|
| POST     | /file | True | Create a record with plain HTML |
| PUT     | /{tid} | True | Modify a record |
| DELETE     | /{tid} | True | Delete a record |
## Event Extraction Flow
We integrate pagination recognition service with api creation system to reach large-scale event extraction, system flow is shown in following figure.
![](https://i.imgur.com/uuFj99g.png)

Code are located under 'queue_app/worker.py', after execute by:
```shell=
python worker.py
```

Program will open connection to RabbitMQ(setting can be customized with **dotenv** package) to receive task from task source queue. (.env details can be accessed in my google shared folder)

### Notice
1. Task message is required to have URL format.
2. Pagination prediction's results will be stored in container 'pageDB_v'.
```shell=
CONTAINER ID        IMAGE                   COMMAND                  CREATED             STATUS              PORTS                                                                                                         NAMES
b86ae6e60415        mongo                   "docker-entrypoint.s…"   4 months ago        Up 44 hours         0.0.0.0:27000->27017/tcp                                                                                      pageDB_v
```
3. After pagination recognition, result task ID will be passed into data etl work queue then finish subsequent process automatically.


## Evaluate Event Extraction

In folder queue_app, you can use 'Event_manual_throw.ipynb' to throw url to event's queue directly, and 'AEStatistic.ipynb' is used to calculate the # of Single/Multi page result and # of legal extractors.
