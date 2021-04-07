import os

import tensorflow as tf
from absl import app as absl_app

import census_dataset
from census_main import Config, build_estimator


def run_evaluate(flags_obj):
    def eval_input_fn():
        test_file = os.path.join(flags_obj.data_dir, census_dataset.EVAL_FILE)
        return census_dataset.input_fn(test_file, 1, False, flags_obj.batch_size)

    model = build_estimator(
        model_dir=flags_obj.model_dir, model_type=flags_obj.model_type,
        model_column_fn=census_dataset.build_model_columns,
        inter_op=flags_obj.inter_op_parallelism_threads,
        intra_op=flags_obj.intra_op_parallelism_threads)
    result = model.evaluate(input_fn=eval_input_fn)
    print(result)


def main(args):
    config = Config()
    current_path = os.path.split(os.path.realpath(__file__))[0] + '/..'
    config.data_dir = current_path + '/census_data'
    config.model_dir = '/tmp/census_model'
    run_evaluate(config)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    absl_app.run(main)
