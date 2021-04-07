# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Train DNN on census income dataset."""

import os

import tensorflow as tf
from absl import app as absl_app

import census_dataset
import wide_deep_run_loop


def build_estimator(model_dir, model_type, model_column_fn, inter_op, intra_op):
    """Build an estimator appropriate for the given model type."""
    wide_columns, deep_columns = model_column_fn()
    hidden_units = [100, 75, 50, 25]

    # Create a tf.estimator.RunConfig to ensure the model is run on CPU, which
    # trains faster than GPU for this model.
    run_config = tf.estimator.RunConfig().replace(
        session_config=tf.ConfigProto(device_count={'GPU': 0},
                                      inter_op_parallelism_threads=inter_op,
                                      intra_op_parallelism_threads=intra_op),
        save_checkpoints_steps=600)

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


def run_census(flags_obj):
    """Construct all necessary functions and call run_loop.

    Args:
      flags_obj: Object containing user specified flags.
    """
    if flags_obj.download_if_missing:
        census_dataset.download(flags_obj.data_dir)

    train_file = os.path.join(flags_obj.data_dir, census_dataset.TRAINING_FILE)
    test_file = os.path.join(flags_obj.data_dir, census_dataset.EVAL_FILE)

    # Train and evaluate the model every `flags.epochs_between_evals` epochs.
    def train_input_fn():
        return census_dataset.input_fn(
            train_file, flags_obj.epochs_between_evals, True, flags_obj.batch_size)

    def eval_input_fn():
        return census_dataset.input_fn(test_file, 1, False, flags_obj.batch_size)

    tensors_to_log = {
        'average_loss': '{loss_prefix}head/truediv',
        'loss': '{loss_prefix}head/weighted_loss/Sum'
    }

    wide_deep_run_loop.run_loop(
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
        model_column_fn=census_dataset.build_model_columns,
        build_estimator_fn=build_estimator,
        flags_obj=flags_obj,
        tensors_to_log=tensors_to_log,
        early_stop=True)


class Config(object):
    def __init__(self):
        self.data_dir = 'census_data/'
        self.batch_size = 32
        self.epochs_between_evals = 1
        self.download_if_missing = True
        self.model_dir = "census_model"
        self.model_type = "wide_and_deep"
        self.inter_op_parallelism_threads = 8
        self.intra_op_parallelism_threads = 8
        self.train_epochs = 320000
        self.export_dir = "census_export"
        self.stop_threshold = 0.7
        self.task_type = ''
        self.task_index = None
        self.max_steps = None
        self.run_mode = 'batch'


def main(_):
    config = Config()
    run_census(config)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    absl_app.run(main)
