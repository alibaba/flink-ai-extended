# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Core run logic for TensorFlow Wide & Deep Tutorial using tf.estimator API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numbers

import ai_flow as af
import tensorflow as tf  # pylint: disable=g-bad-import-order

LOSS_PREFIX = {'wide': 'linear/', 'deep': 'dnn/'}


def export_model(model, model_type, export_dir, model_column_fn):
    """Export to SavedModel format.

    Args:
      model: Estimator object
      model_type: string indicating model type. "wide", "deep" or "wide_deep"
      export_dir: directory to export the model.
      model_column_fn: Function to generate model feature columns.
    """
    wide_columns, deep_columns = model_column_fn()
    if model_type == 'wide':
        columns = wide_columns
    elif model_type == 'deep':
        columns = deep_columns
    else:
        columns = wide_columns + deep_columns
    feature_spec = tf.feature_column.make_parse_example_spec(columns)
    example_input_fn = (
        tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec))
    return str(model.export_savedmodel(export_dir, example_input_fn,
                                       strip_default_attrs=True), encoding="utf-8")


def past_stop_threshold(stop_threshold, eval_metric):
    """Return a boolean representing whether a model should be stopped.

    Args:
      stop_threshold: float, the threshold above which a model should stop
        training.
      eval_metric: float, the current value of the relevant metric to check.

    Returns:
      True if training should stop, False otherwise.

    Raises:
      ValueError: if either stop_threshold or eval_metric is not a number
    """
    if stop_threshold is None:
        return False

    if not isinstance(stop_threshold, numbers.Number):
        raise ValueError("Threshold for checking stop conditions must be a number.")
    if not isinstance(eval_metric, numbers.Number):
        raise ValueError("Eval metric being checked against stop conditions "
                         "must be a number.")

    if eval_metric >= stop_threshold:
        tf.logging.info(
            "Stop threshold of {} was passed with metric value {}.".format(
                stop_threshold, eval_metric))
        return True

    return False


class ExportCheckpointSaverListener(tf.estimator.CheckpointSaverListener):
    def __init__(self, estimator, model_type, export_dir, model_column_fn):
        self.estimator = estimator
        self.model_type = model_type
        self.export_dir = export_dir
        self.model_column_fn = model_column_fn

    def begin(self):
        # You can add ops to the graph here.
        print('Starting the session.')

    def before_save(self, session, global_step_value):
        print('About to write a checkpoint')

    def after_save(self, session, global_step_value):
        real_path = export_model(self.estimator, self.model_type, self.export_dir,
                                 self.model_column_fn)
        print("final {}".format(real_path))
        # todo copy checkpoint
        deploy_mv = af.get_deployed_model_version('wide_and_deep')
        if deploy_mv is not None:
            af.update_model_version(model_name='wide_and_deep', model_version=deploy_mv.version,
                                    current_stage=af.ModelVersionStage.GENERATED)
        af.register_model_version(model='wide_and_deep', model_path='|' + real_path,
                                  current_stage=af.ModelVersionStage.DEPLOYED)

    def end(self, session, global_step_value):
        print('Done with the session.')


def run_loop(train_input_fn, eval_input_fn, model_column_fn,
             build_estimator_fn, flags_obj, tensors_to_log, early_stop=False):
    """Define training loop."""
    model = build_estimator_fn(
        model_dir=flags_obj.model_dir, model_type=flags_obj.model_type,
        model_column_fn=model_column_fn,
        inter_op=flags_obj.inter_op_parallelism_threads,
        intra_op=flags_obj.intra_op_parallelism_threads)

    train_hooks = []
    ll = ExportCheckpointSaverListener(model, flags_obj.model_type, flags_obj.export_dir,
                                       model_column_fn)
    # Train and evaluate the model every `flags.epochs_between_evals` epochs.
    for n in range(flags_obj.train_epochs // flags_obj.epochs_between_evals):
        model.train(input_fn=train_input_fn, hooks=train_hooks, saving_listeners=[ll])

        results = model.evaluate(input_fn=eval_input_fn)

        # Display evaluation metrics
        tf.logging.info('Results at epoch %d / %d',
                        (n + 1) * flags_obj.epochs_between_evals,
                        flags_obj.train_epochs)
        tf.logging.info('-' * 60)

        for key in sorted(results):
            tf.logging.info('%s: %s' % (key, results[key]))

        if early_stop and past_stop_threshold(
                flags_obj.stop_threshold, results['accuracy']):
            break

    # Export the model
    if flags_obj.export_dir is not None:
        export_model(model, flags_obj.model_type, flags_obj.export_dir,
                     model_column_fn)
