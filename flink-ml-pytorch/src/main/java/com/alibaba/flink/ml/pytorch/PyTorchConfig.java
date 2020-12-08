/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.alibaba.flink.ml.pytorch;

import com.alibaba.flink.ml.cluster.MLConfig;
import com.alibaba.flink.ml.cluster.role.WorkerRole;

import java.util.HashMap;
import java.util.Map;

/**
 * PyTorch machine learning cluster configuration.
 */
public class PyTorchConfig {
	private MLConfig mlConfig;

	/**
	 * create PyTorch machine learning cluster configuration.
	 * @param world_size the number of PyTorch cluster worker number.
	 * @param properties cluster configuration properties.
	 * @param pythonFiles PyTorch job run python scripts.
	 * @param funName PyTorch job script main function.
	 * @param envPath python virtual environment address.
	 */
	public PyTorchConfig(int world_size, Map<String, String> properties, String[] pythonFiles,
			String funName, String envPath) {
		Map<String, Integer> jobNum = new HashMap<>();
		jobNum.put(new WorkerRole().name(), world_size);
		this.mlConfig = new MLConfig(jobNum, properties, pythonFiles, funName, envPath);
	}

	/**
	 * create PyTorch machine learning cluster configuration.
	 * @param world_size the number of PyTorch cluster worker number.
	 * @param properties cluster configuration properties.
	 * @param pythonFiles PyTorch job run python scripts.
	 * @param funName PyTorch job script main function.
	 * @param envPath python virtual environment address.
	 */
	public PyTorchConfig(int world_size, Map<String, String> properties, String pythonFiles,
			String funName, String envPath) {
		Map<String, Integer> jobNum = new HashMap<>();
		jobNum.put(new WorkerRole().name(), world_size);
		this.mlConfig = new MLConfig(jobNum, properties, pythonFiles, funName, envPath);
	}

	/**
	 * @return machine learning cluster configuration.
	 */
	public MLConfig getMlConfig() {
		return mlConfig;
	}
}

