import json
import os
import time

import ai_flow as af
import tensorflow as tf
from absl import app as absl_app
from flink_ml_tensorflow.tensorflow_context import TFContext

import census_dataset
from census_main import Config
from wide_deep_run_loop import export_model, ExportCheckpointSaverListener


def build_estimator(model_dir, model_type, model_column_fn, inter_op, intra_op):
    """Build an estimator appropriate for the given model type."""
    wide_columns, deep_columns = model_column_fn()
    hidden_units = [100, 75, 50, 25]

    # Create a tf.estimator.RunConfig to ensure the model is run on CPU, which
    # trains faster than GPU for this model.
    run_config = tf.estimator.RunConfig().replace(
        keep_checkpoint_max=2,
        save_checkpoints_secs=20,
        session_config=tf.ConfigProto(device_count={'GPU': 0},
                                      inter_op_parallelism_threads=inter_op,
                                      intra_op_parallelism_threads=intra_op))

    if model_type == 'wide':
        return tf.estimator.LinearClassifier(
            model_dir=model_dir,
            feature_columns=wide_columns,
            config=run_config)
    elif model_type == 'deep':
        return tf.estimator.DNNClassifier(
            model_dir=model_dir,
            feature_columns=deep_columns,
            hidden_units=hidden_units,
            config=run_config)
    else:
        return tf.estimator.DNNLinearCombinedClassifier(
            model_dir=model_dir,
            linear_feature_columns=wide_columns,
            dnn_feature_columns=deep_columns,
            dnn_hidden_units=hidden_units,
            config=run_config)


def run_census(flags_obj, input_fun):
    """Construct all necessary functions and call run_loop.

    Args:
      flags_obj: Object containing user specified flags.
    """
    if flags_obj.download_if_missing:
        census_dataset.download(flags_obj.data_dir)

    # Create a cluster from the parameter server and worker hosts.
    config = json.loads(os.environ['TF_CONFIG'])
    cluster = tf.train.ClusterSpec(config['cluster'])
    # Create and start a server for the local task.
    server = tf.train.Server(cluster,
                             job_name=flags_obj.task_type,
                             task_index=int(flags_obj.task_index))
    if 'ps' == flags_obj.task_type:
        while True:
            time.sleep(10)

    """Define training loop."""
    af.set_project_config_file(os.path.dirname(__file__) + '/project.yaml')

    if 'batch' == flags_obj.run_mode:
        version = round(time.time())
        model_path = str(flags_obj.model_dir + '/' + str(version))
    else:
        model_path = af.get_deployed_model_version('wide_and_deep').model_path.split('|')[0]
    print(model_path)
    model = build_estimator(
        model_dir=model_path, model_type=flags_obj.model_type,
        model_column_fn=census_dataset.build_model_columns,
        inter_op=flags_obj.inter_op_parallelism_threads,
        intra_op=flags_obj.intra_op_parallelism_threads)
    ll = []
    if 'stream' == flags_obj.run_mode:
        ll = [ExportCheckpointSaverListener(model, flags_obj.model_type, flags_obj.export_dir,
                                            census_dataset.build_model_columns)]
    train_hooks = []
    model.train(input_fn=input_fun, hooks=train_hooks, max_steps=flags_obj.max_steps, saving_listeners=ll)

    # Export the model
    if flags_obj.export_dir is not None and 'batch' == flags_obj.run_mode and flags_obj.task_type == 'chief':
        export_path = export_model(model, flags_obj.model_type, flags_obj.export_dir,
                                   census_dataset.build_model_columns)
        print(model_path)
        print(export_path)
        deploy_mv = af.get_deployed_model_version('wide_and_deep')
        if deploy_mv is not None:
            af.update_model_version(model_name='wide_and_deep', model_version=deploy_mv.version,
                                    current_stage=af.ModelVersionStage.GENERATED)
        af.register_model_version(model='wide_and_deep', model_path=model_path + '|' + export_path,
                                  current_stage=af.ModelVersionStage.DEPLOYED)


def map_func(context):
    tf_context = TFContext(context)
    job_name = tf_context.get_role_name()
    index = tf_context.get_index()
    tf_context.export_estimator_cluster()
    print("job name:" + job_name)
    print("current index:" + str(index))
    tf_config = json.loads(os.environ['TF_CONFIG'])
    config = Config()
    config.max_steps = 500
    config.data_dir = '/tmp/census_data'
    config.model_dir = '/tmp/census_model'
    config.export_dir = '/tmp/census_export'
    config.task_type = tf_config['task']['type']
    config.task_index = tf_config['task']['index']
    config.run_mode = 'batch'
    train_file = os.path.join(config.data_dir, census_dataset.TRAINING_FILE)

    def train_input_fn():
        return census_dataset.input_fn(
            train_file, -1, True, config.batch_size)

    run_census(config, train_input_fn)


def stream_map_func(context):
    tf_context = TFContext(context)

    def flink_input_fn(batch_size):
        """Generate an input function for the Estimator."""

        def parse_csv(value):
            columns = tf.decode_csv(value, record_defaults=census_dataset._CSV_COLUMN_DEFAULTS)
            features = dict(zip(census_dataset._CSV_COLUMNS, columns))
            labels = features.pop('income_bracket')
            classes = tf.equal(labels, '>50K')  # binary classification
            return features, classes

        dataset = tf_context.flink_stream_dataset()
        dataset = dataset.map(parse_csv, num_parallel_calls=5)
        dataset = dataset.repeat(1)
        dataset = dataset.batch(batch_size)
        return dataset

    def flink_train_input_fn():
        return flink_input_fn(config.batch_size)

    job_name = tf_context.get_role_name()
    index = tf_context.get_index()
    tf_context.export_estimator_cluster()
    print("job name:" + job_name)
    print("current index:" + str(index))
    tf_config = json.loads(os.environ['TF_CONFIG'])
    config = Config()
    config.run_mode = 'stream'
    config.max_steps = None
    config.data_dir = '/tmp/census_data'
    config.model_dir = '/tmp/census_model'
    config.export_dir = '/tmp/census_export'
    config.task_type = tf_config['task']['type']
    config.task_index = tf_config['task']['index']
    run_census(config, flink_train_input_fn)


def main(args):
    config = Config()
    current_path = os.path.split(os.path.realpath(__file__))[0]
    config.data_dir = current_path + '/census_data'
    config.task_type = args[1]
    config.task_index = args[2]
    os.environ['TF_CONFIG'] = json.dumps({
        'cluster': {
            'chief': ['localhost:1111'],
            'worker': ['localhost:1112'],
            'ps': ['localhost:1113']
        },
        'task': {'type': config.task_type, 'index': int(config.task_index)}
    })
    run_census(config)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    absl_app.run(main)
