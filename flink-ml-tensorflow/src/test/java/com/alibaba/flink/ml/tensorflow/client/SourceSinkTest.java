package com.alibaba.flink.ml.tensorflow.client;

import com.alibaba.flink.ml.operator.util.DataTypes;
import com.alibaba.flink.ml.tensorflow.coding.ExampleCoding;
import com.alibaba.flink.ml.tensorflow.coding.ExampleCodingConfig;
import com.alibaba.flink.ml.tensorflow.util.TFConstants;
import com.alibaba.flink.ml.util.MLConstants;
import com.alibaba.flink.ml.util.TestUtil;
import org.apache.curator.test.TestingServer;
import org.apache.flink.api.common.restartstrategy.RestartStrategies;
import org.apache.flink.api.common.state.ListState;
import org.apache.flink.api.common.state.ListStateDescriptor;
import org.apache.flink.api.common.typeinfo.BasicTypeInfo;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.runtime.state.FunctionInitializationContext;
import org.apache.flink.runtime.state.FunctionSnapshotContext;
import org.apache.flink.streaming.api.checkpoint.CheckpointedFunction;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.source.SourceFunction;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.TableSchema;
import org.apache.flink.table.api.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.sql.Timestamp;
import java.util.HashMap;
import java.util.Map;

public class SourceSinkTest {
    private static final String ZookeeperConn = "127.0.0.1:2181";
    private static final String[] Scripts = {TestUtil.getProjectRootPath() + "/flink-ml-tensorflow/src/test/python/source_sink.py"};
    private static final int WorkerNum = 2;
    private static final int PsNum = 0;

    @Test
    public void testSourceSink() throws Exception {
        TestingServer server = new TestingServer(2181, true);
        StreamExecutionEnvironment streamEnv = StreamExecutionEnvironment.createLocalEnvironment(1);
        streamEnv.setRestartStrategy(RestartStrategies.noRestart());

        DataStream<Row> sourceStream = streamEnv.addSource(
                new DummyTimedSource(16, 5), new RowTypeInfo(Types.STRING)).setParallelism(1);
        StreamTableEnvironment tableEnv = StreamTableEnvironment.create(streamEnv);
        Table input = tableEnv.fromDataStream(sourceStream, "input");
        TFConfig config = createTFConfig("test_source_sink");
        TableSchema inputSchema = new TableSchema(new String[]{"input"}, new TypeInformation[]{BasicTypeInfo.STRING_TYPE_INFO});
        TableSchema outputSchema = new TableSchema(new String[]{"output"}, new TypeInformation[]{BasicTypeInfo.STRING_TYPE_INFO});

        String strInput = ExampleCodingConfig.createExampleConfigStr(new String[]{"input"}, new DataTypes[]{DataTypes.STRING},
                ExampleCodingConfig.ObjectType.ROW, Row.class);
        config.getProperties().put(TFConstants.INPUT_TF_EXAMPLE_CONFIG, strInput);
        config.getProperties().put(MLConstants.ENCODING_CLASS, ExampleCoding.class.getCanonicalName());
        String strOutput = ExampleCodingConfig.createExampleConfigStr(new String[]{"output"}, new DataTypes[]{DataTypes.STRING},
                ExampleCodingConfig.ObjectType.ROW, Row.class);
        config.getProperties().put(TFConstants.OUTPUT_TF_EXAMPLE_CONFIG, strOutput);
        config.getProperties().put(MLConstants.DECODING_CLASS, ExampleCoding.class.getCanonicalName());

        Table output = TFUtils.inference(streamEnv, tableEnv, input, config, outputSchema);
        tableEnv.toAppendStream(output, Row.class)
                .map(r -> {
                    return "[Sink][" + new Timestamp(System.currentTimeMillis()) + "]finish " + r.getField(0) + "\n";
                }).print().setParallelism(1);
        streamEnv.execute();
        server.stop();
    }

    private TFConfig createTFConfig(String mapFunc) {
        Map<String, String> prop = new HashMap<>();
        prop.put(MLConstants.CONFIG_STORAGE_TYPE, MLConstants.STORAGE_ZOOKEEPER);
        prop.put(MLConstants.CONFIG_ZOOKEEPER_CONNECT_STR, ZookeeperConn);
        return new TFConfig(WorkerNum, PsNum, prop, Scripts, mapFunc, null);
    }

    private static class DummyTimedSource implements SourceFunction<Row>, CheckpointedFunction {
        public static final Logger LOG = LoggerFactory.getLogger(DummyTimedSource.class);
        private long count = 0L;
        private long MAX_COUNT;
        private long INTERVAL;
        private volatile boolean isRunning = true;

        private transient ListState<Long> checkpointedCount;

        public DummyTimedSource(long maxCount, long interval) {
            this.MAX_COUNT = maxCount;
            this.INTERVAL = interval;
        }

        @Override
        public void run(SourceContext<Row> ctx) throws Exception {
            while (isRunning && count < MAX_COUNT) {
                // this synchronized block ensures that state checkpointing,
                // internal state updates and emission of elements are an atomic operation
                synchronized (ctx.getCheckpointLock()) {
                    Row row = new Row(1);
                    row.setField(0, String.format("data-%d", count));
                    System.out.println("[Source][" + new Timestamp(System.currentTimeMillis()) + "]produce " + row.getField(0));
                    ctx.collect(row);
                    count++;
                    Thread.sleep(INTERVAL * 1000);
                }
            }
        }

        @Override
        public void cancel() {
            isRunning = false;
        }

        @Override
        public void snapshotState(FunctionSnapshotContext context) throws Exception {
            this.checkpointedCount.clear();
            this.checkpointedCount.add(count);
        }

        @Override
        public void initializeState(FunctionInitializationContext context) throws Exception {
            this.checkpointedCount = context
                    .getOperatorStateStore()
                    .getListState(new ListStateDescriptor<>("count", Long.class));

            if (context.isRestored()) {
                for (Long count : this.checkpointedCount.get()) {
                    this.count = count;
                }
            }
        }
    }
}
