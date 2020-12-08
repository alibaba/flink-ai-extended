package com.alibaba.flink.ml.tensorflow.cluster;

import com.alibaba.flink.ml.tensorflow.client.TFConfig;
import com.alibaba.flink.ml.tensorflow.client.TFUtils;
import com.alibaba.flink.ml.util.MLConstants;
import com.alibaba.flink.ml.util.TestUtil;
import org.apache.curator.test.TestingServer;
import org.apache.flink.api.common.JobExecutionResult;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.HashMap;
import java.util.Map;

public class PythonFailedTest {
    private static TestingServer server;
    private static final String pythonPath = TestUtil.getProjectRootPath() + "/flink-ml-tensorflow/src/test/python/";
    private static final String t = pythonPath + "t.py";

    @Before
    public void setUp() throws Exception {
        server = new TestingServer(2181, true);
    }

    @After
    public void tearDown() throws Exception {
        server.stop();
    }

    @Test
    public void failTest() throws Exception{
        StreamExecutionEnvironment streamEnv = StreamExecutionEnvironment.getExecutionEnvironment();
        Map<String, String> prop = new HashMap<>();
//        prop.put(MLConstants.START_WITH_STARTUP,"false");
        TFConfig config = new TFConfig(1, 0, null, t, "main", null);

        TFUtils.train(streamEnv, null, config);
        JobExecutionResult result = streamEnv.execute();
    }
}
