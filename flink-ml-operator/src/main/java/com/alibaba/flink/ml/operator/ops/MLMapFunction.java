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

package com.alibaba.flink.ml.operator.ops;

import com.alibaba.flink.ml.cluster.ExecutionMode;
import com.alibaba.flink.ml.cluster.MLConfig;
import com.alibaba.flink.ml.cluster.node.MLContext;
import com.alibaba.flink.ml.cluster.rpc.NodeServer;
import com.alibaba.flink.ml.data.DataExchange;
import com.alibaba.flink.ml.operator.util.PythonFileUtil;
import com.alibaba.flink.ml.cluster.role.BaseRole;

import org.apache.flink.api.common.functions.RuntimeContext;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.util.Collector;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Closeable;
import java.io.IOException;
import java.io.InterruptedIOException;
import java.io.Serializable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

/**
 * machine learning node has input and output,MLMapFunction is a util function to help create MLFlatMapOp class object.
 * @param <IN> machine learning node input class.
 * @param <OUT> machine learning node output class.
 */
public class MLMapFunction<IN, OUT> implements Closeable, Serializable {
	private BaseRole role;
	private MLConfig config;
	private TypeInformation<IN> inTI;
	private TypeInformation<OUT> outTI;
	private MLContext mlContext;
	private ExecutorService serverService;
    private ExecutorService collectorService;
	private Future collectorTaskFuture;
	private Future nodeServerTaskFuture;
	private ExecutionMode mode;
	private transient DataExchange<IN, OUT> dataExchange;
	private volatile Collector<OUT> collector = null;

	private static final Logger LOG = LoggerFactory.getLogger(MLMapFunction.class);

	public MLMapFunction(ExecutionMode mode, BaseRole role, MLConfig config, TypeInformation<IN> inTI,
			TypeInformation<OUT> outTI) {
		this.mode = mode;
		this.role = role;
		this.config = config;
		this.outTI = outTI;
		this.inTI = inTI;
	}

	/**
	 * create machine learning node and data exchange object.
	 * @param runtimeContext flink operator RuntimeContext.
	 * @throws Exception
	 */
	public void open(RuntimeContext runtimeContext) throws Exception {
		ResourcesUtils.parseGpuInfo(runtimeContext, config);
		mlContext = new MLContext(mode, config, role.name(), runtimeContext.getIndexOfThisSubtask(),
				config.getEnvPath(), null);
		PythonFileUtil.preparePythonFilesForExec(runtimeContext, mlContext);

		dataExchange = new DataExchange<>(mlContext);

		serverService = Executors.newFixedThreadPool(1, r -> {
			Thread t = new Thread(r);
			t.setDaemon(true);
			t.setName("NodeServer_" + mlContext.getIdentity());
			return t;
		});
		nodeServerTaskFuture = serverService.submit(new NodeServer(mlContext, role.name()));

		collectorService = Executors.newFixedThreadPool(1, r -> {
			Thread t = new Thread(r);
			t.setDaemon(true);
			t.setName("ResultCollector_" + mlContext.getIdentity());
			return t;
		});
		collectorTaskFuture = collectorService.submit(() -> {
			while (true) {
				if (collector == null) {
					try {
						LOG.info("collector is still null, sleep 1 second...");
						Thread.sleep(1000);
					} catch (InterruptedException e) {
						e.printStackTrace();
					}
				} else {
					break;
				}
			}
			drainRead(collector, true);
		});

		System.out.println("start:" + mlContext.getRoleName() + " index:" + mlContext.getIndex());
	}

	/**
	 * stop machine learning node and resource.
	 */
	@Override
	public void close() {
		if (mlContext != null && mlContext.getOutputQueue() != null) {
			mlContext.getOutputQueue().markFinished();
		}

		// wait for tf thread finish
		closeCollector();
		closeNodeServer();


		LOG.info("Records output: " + dataExchange.getReadRecords());

		if (mlContext != null) {
			try {
				mlContext.close();
			} catch (IOException e) {
				LOG.error("Fail to close mlContext.", e);
			}
			mlContext = null;
		}
	}

	/**
	 * stop node server
	 */
	private void closeNodeServer() {
		LOG.info("try to close node server");
		if (serverService == null) {
			LOG.info("node server has been closed");
			return;
		}

		try {
			nodeServerTaskFuture.get();
		} catch (InterruptedException | ExecutionException e) {
			LOG.warn("failed to join node server task" + " index:" + mlContext.getIndex());
			e.printStackTrace();
		} finally {
			serverService.shutdown();
			try {
				if (!serverService.awaitTermination(10, TimeUnit.SECONDS)) {
					LOG.warn("failed to shutdown node server" + " index:" + mlContext.getIndex());
				}
			} catch (InterruptedException e) {
				e.printStackTrace();
				LOG.warn("failed to shutdown node server" + " index:" + mlContext.getIndex());
			} finally {
				if (!serverService.isShutdown()) {
					serverService.shutdownNow();
				}
			}
			nodeServerTaskFuture = null;
			serverService = null;
		}
	}

	/**
	 * stop result collector
	 */
	private void closeCollector() {
		LOG.info("try to close collector service");
		if (collectorService == null) {
			LOG.info("collector service has been closed");
			return;
		}

		try {
			collectorTaskFuture.get();
		} catch (InterruptedException | ExecutionException e) {
			LOG.warn("failed to join collector task" + " index:" + mlContext.getIndex());
			e.printStackTrace();
		} finally {
			collectorService.shutdown();
			try {
				if (!collectorService.awaitTermination(10, TimeUnit.SECONDS)) {
					LOG.warn("failed to shutdown result collector" + " index:" + mlContext.getIndex());
				}
			} catch (InterruptedException e) {
				e.printStackTrace();
				LOG.warn("failed to shutdown result collector" + " index:" + mlContext.getIndex());
			} finally {
				if (!collectorService.isShutdown()) {
					collectorService.shutdownNow();
				}
			}
			collectorTaskFuture = null;
			collectorService = null;
		}
	}

	/**
	 * process input data and collect results.
	 * @param value input object.
	 * @param out output result.
	 * @throws Exception
	 */
	void flatMap(IN value, Collector<OUT> out) throws Exception {
		collector = out;

		//put the read & write in a loop to avoid dead lock between write queue and read queue.
		boolean writeSuccess = false;
		do {
//			drainRead(out, false);

			writeSuccess = dataExchange.write(value);
			if (!writeSuccess) {
				Thread.yield();
			}
		} while (!writeSuccess);
	}

	public TypeInformation<OUT> getProducedType() {
		return outTI;
	}

	private void drainRead(Collector<OUT> out, boolean readUntilEOF) {
		while (true) {
			try {
				Object r = dataExchange.read(readUntilEOF);
				if (r != null) {
					out.collect((OUT) r);
				} else {
					break;
				}
			} catch (InterruptedIOException iioe) {
				LOG.info("{} Reading from is interrupted, canceling the server", mlContext.getIdentity());
				closeCollector();
				closeNodeServer();
			} catch (IOException e) {
				LOG.error("Fail to read data from python.", e);
			}
		}
	}
}
