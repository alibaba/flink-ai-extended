#!/usr/bin/env bash
##
## Licensed to the Apache Software Foundation (ASF) under one
## or more contributor license agreements.  See the NOTICE file
## distributed with this work for additional information
## regarding copyright ownership.  The ASF licenses this file
## to you under the Apache License, Version 2.0 (the
## "License"); you may not use this file except in compliance
## with the License.  You may obtain a copy of the License at
##
##   http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing,
## software distributed under the License is distributed on an
## "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
## KIND, either express or implied.  See the License for the
## specific language governing permissions and limitations
## under the License.
##
set -e
export AIRFLOW_HOME=${AIRFLOW_HOME:-~/airflow}
export AIFLOW_PID_DIR=${AIFLOW_PID_DIR:-/tmp}

MYSQL_CONN=$1
if [ ! ${MYSQL_CONN} ]; then
  echo "Error: you need to provide a mysql database to start the airflow and notification service."
  exit 1
fi

if [[ ! -f "${AIRFLOW_HOME}/airflow.cfg" ]] ; then
    mkdir ${AIRFLOW_HOME} >/dev/null 2>&1 || true

    CURRENT_DIR=$(pwd)
    cd ${AIRFLOW_HOME}

    # create the configuration file
    airflow db init >/dev/null 2>&1 || true
    mv airflow.cfg airflow.cfg.tmpl
    awk "{gsub(\"sql_alchemy_conn = sqlite:///${AIRFLOW_HOME}/airflow.db\", \"sql_alchemy_conn = ${MYSQL_CONN}\"); \
        gsub(\"load_examples = True\", \"load_examples = False\"); \
        gsub(\"load_default_connections = True\", \"load_default_connections = False\"); \
        gsub(\"dag_dir_list_interval = 300\", \"dag_dir_list_interval = 3\"); \
        gsub(\"executor = SequentialExecutor\", \"executor = LocalExecutor\"); \
        gsub(\"dags_are_paused_at_creation = True\", \"dags_are_paused_at_creation = False\"); \
        gsub(\"# mp_start_method =\", \"mp_start_method = forkserver\"); \
        gsub(\"execute_tasks_new_python_interpreter = False\", \"execute_tasks_new_python_interpreter = True\"); \
        gsub(\"min_serialized_dag_update_interval = 30\", \"min_serialized_dag_update_interval = 0\"); \
        print \$0}" airflow.cfg.tmpl > airflow.cfg
    rm airflow.cfg.tmpl >/dev/null 2>&1 || true
    # prepare the database
    airflow db reset -y

    cd ${CURRENT_DIR}
fi

# create a default Admin user for airflow
airflow users create \
    --username admin \
    --password admin \
    --firstname admin \
    --lastname admin \
    --role Admin \
    --email admin@example.org

AIRFLOW_DEPLOY_PATH="${AIRFLOW_HOME}/airflow_deploy"
AIFLOW_LOG_DIR="${AIRFLOW_HOME}/logs"

# create the pending directory if not exists
mkdir ${AIRFLOW_DEPLOY_PATH} >/dev/null 2>&1 || true

if [ -d "${AIFLOW_LOG_DIR}" ]; then
  time=$(date "+%Y%m%d-%H%M%S")
  mv "${AIFLOW_LOG_DIR}" "${AIFLOW_LOG_DIR}.${time}"
fi

mkdir ${AIFLOW_LOG_DIR} >/dev/null 2>&1 || true

# start notification service
start_notification_service.py --database-conn=${MYSQL_CONN} > ${AIFLOW_LOG_DIR}/notification_service.log 2>&1 &
echo $! > ${AIFLOW_PID_DIR}/notification_service.pid

# start airflow scheduler and web server
airflow event_scheduler --subdir=${AIRFLOW_DEPLOY_PATH} > ${AIFLOW_LOG_DIR}/scheduler.log 2>&1 &
echo $! > ${AIFLOW_PID_DIR}/scheduler.pid
airflow webserver -p 8080 > ${AIFLOW_LOG_DIR}/web.log 2>&1 &
echo $! > ${AIFLOW_PID_DIR}/web.pid
start_aiflow.py > ${AIFLOW_LOG_DIR}/master_server.log 2>&1 &
echo $! > ${AIFLOW_PID_DIR}/master_server.pid

echo "Notification service log: ${AIFLOW_LOG_DIR}/notification_service.log"
echo "Notification service pid: $(cat ${AIFLOW_PID_DIR}/notification_service.pid)"
echo "Scheduler log: ${AIFLOW_LOG_DIR}/scheduler.log"
echo "Scheduler pid: $(cat ${AIFLOW_PID_DIR}/scheduler.pid)"
echo "Web Server log: ${AIFLOW_LOG_DIR}/web.log"
echo "Web Server pid: $(cat ${AIFLOW_PID_DIR}/web.pid)"
echo "Master Server log: ${AIFLOW_LOG_DIR}/master_server.log"
echo "Master Server pid: $(cat ${AIFLOW_PID_DIR}/master_server.pid)"
echo "Airflow deploy path: ${AIRFLOW_DEPLOY_PATH}"
echo "Visit http://127.0.0.1:8080/ to access the airflow web server."
