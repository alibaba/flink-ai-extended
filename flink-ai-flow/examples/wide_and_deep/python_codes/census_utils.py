import os
import shutil
import time


# from pyhdfs import HdfsClient
#
#
# def hdfs_copy_checkpoint_to_model_path(hosts, train_dir, model_base_path):
#     client = HdfsClient(hosts=hosts)
#     if not client.exists(model_base_path):
#         client.mkdirs(model_base_path)
#         tmp = '/tmp/' + str(uuid.uuid4())
#         client.copy_to_local(train_dir, tmp)
#         client.copy_from_local(tmp, model_base_path)


def copy_checkpoint_to_model_path(train_dir, model_base_path):
    version = round(time.time())
    os.makedirs(model_base_path, exist_ok=True)
    dist_path = model_base_path + '/' + version
    shutil.copytree(train_dir, dist_path)
    return dist_path


if __name__ == "__main__":
    copy_checkpoint_to_model_path('../census_model', '../mmm')
    # hdfs_copy_checkpoint_to_model_path('localhost:50070', '/demo', '/xx/dd')
