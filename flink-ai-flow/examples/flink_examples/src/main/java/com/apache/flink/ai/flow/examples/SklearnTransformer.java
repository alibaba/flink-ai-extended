package com.apache.flink.ai.flow.examples;


import org.apache.flink.ml.api.core.Transformer;
import org.apache.flink.ml.api.misc.param.Params;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.TableEnvironment;


public class SklearnTransformer implements Transformer<SklearnTransformer> {

    private Params params = new Params();

    @Override
    public Table transform(TableEnvironment tEnv, Table input) {
        return input.select("facidity, vacidity, cacid, rsugar, tdioxide, density, pH, sulphates, alcohol, quality");
    }

    @Override
    public Params getParams() {
        return params;
    }
}
