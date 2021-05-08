#!/usr/bin/env bash
##
## Licensed to the Apache Software Foundation (ASF) under one
## or more contributor license agreements.  See the NOTICE file
## distributed with this work for additional information
## regarding copyright ownership.  The ASF licenses this file
## to you under the Apache License, Version 2.0 (the
## "License"); you may not use this file except in compliance
## with the License.  You may obtain a copy of the License at
##
##   http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing,
## software distributed under the License is distributed on an
## "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
## KIND, either express or implied.  See the License for the
## specific language governing permissions and limitations
## under the License.
##
set -e

#generate go file
protoc -I/usr/local/include -I. \
  -I$GOPATH/src \
  -I$GOPATH/src/github.com/grpc-ecosystem/grpc-gateway/third_party/googleapis \
  --go_out=Mgoogle/api/annotations.proto=github.com/grpc-ecosystem/grpc-gateway/third_party/googleapis/google/api,plugins=grpc:../go \
  message.proto

protoc -I/usr/local/include -I. \
  -I$GOPATH/src \
  -I$GOPATH/src/github.com/grpc-ecosystem/grpc-gateway/third_party/googleapis \
  --go_out=Mgoogle/api/annotations.proto=github.com/grpc-ecosystem/grpc-gateway/third_party/googleapis/google/api,plugins=grpc:../go \
  metadata_service.proto

protoc -I/usr/local/include -I. \
  -I$GOPATH/src -I$GOPATH/src/github.com/grpc-ecosystem/grpc-gateway/third_party/googleapis \
  --grpc-gateway_out=logtostderr=true:../go \
  metadata_service.proto

protoc -I/usr/local/include -I. \
  -I$GOPATH/src \
  -I$GOPATH/src/github.com/grpc-ecosystem/grpc-gateway/third_party/googleapis \
  --go_out=Mgoogle/api/annotations.proto=github.com/grpc-ecosystem/grpc-gateway/third_party/googleapis/google/api,plugins=grpc:../go \
  model_center_service.proto

protoc -I/usr/local/include -I. \
  -I$GOPATH/src -I$GOPATH/src/github.com/grpc-ecosystem/grpc-gateway/third_party/googleapis \
  --grpc-gateway_out=logtostderr=true:../go \
  model_center_service.proto

protoc -I/usr/local/include -I. \
  -I$GOPATH/src \
  -I$GOPATH/src/github.com/grpc-ecosystem/grpc-gateway/third_party/googleapis \
  --go_out=Mgoogle/api/annotations.proto=github.com/grpc-ecosystem/grpc-gateway/third_party/googleapis/google/api,plugins=grpc:../go \
  deploy_service.proto

protoc -I/usr/local/include -I. \
  -I$GOPATH/src -I$GOPATH/src/github.com/grpc-ecosystem/grpc-gateway/third_party/googleapis \
  --grpc-gateway_out=logtostderr=true:../go \
  deploy_service.proto

protoc -I/usr/local/include -I. \
  -I$GOPATH/src \
  -I$GOPATH/src/github.com/grpc-ecosystem/grpc-gateway/third_party/googleapis \
  --go_out=Mgoogle/api/annotations.proto=github.com/grpc-ecosystem/grpc-gateway/third_party/googleapis/google/api,plugins=grpc:../go \
  scheduling_service.proto

protoc -I/usr/local/include -I. \
  -I$GOPATH/src -I$GOPATH/src/github.com/grpc-ecosystem/grpc-gateway/third_party/googleapis \
  --grpc-gateway_out=logtostderr=true:../go \
  scheduling_service.proto

protoc -I/usr/local/include -I. \
  -I$GOPATH/src \
  -I$GOPATH/src/github.com/grpc-ecosystem/grpc-gateway/third_party/googleapis \
  --go_out=Mgoogle/api/annotations.proto=github.com/grpc-ecosystem/grpc-gateway/third_party/googleapis/google/api,plugins=grpc:../go \
  metric_service.proto

protoc -I/usr/local/include -I. \
  -I$GOPATH/src -I$GOPATH/src/github.com/grpc-ecosystem/grpc-gateway/third_party/googleapis \
  --grpc-gateway_out=logtostderr=true:../go \
  metric_service.proto

#generate java file
protoc -I/usr/local/include -I. \
  -I$GOPATH/src -I$GOPATH/src/github.com/grpc-ecosystem/grpc-gateway/third_party/googleapis \
  --java_out=../../../java/client/src/main/java \
  --proto_path=. \
  message.proto

protoc -I/usr/local/include -I. \
  -I$GOPATH/src -I$GOPATH/src/github.com/grpc-ecosystem/grpc-gateway/third_party/googleapis \
  --java_out=../../../java/client/src/main/java \
  --proto_path=. \
  metadata_service.proto

protoc -I/usr/local/include -I. \
  -I$GOPATH/src -I$GOPATH/src/github.com/grpc-ecosystem/grpc-gateway/third_party/googleapis \
  --plugin=protoc-gen-grpc-java \
  --grpc-java_out=../../../java/client/src/main/java \
  --proto_path=. \
  metadata_service.proto

protoc -I/usr/local/include -I. \
  -I$GOPATH/src -I$GOPATH/src/github.com/grpc-ecosystem/grpc-gateway/third_party/googleapis \
  --java_out=../../../java/client/src/main/java \
  --proto_path=. \
  model_center_service.proto

protoc -I/usr/local/include -I. \
  -I$GOPATH/src -I$GOPATH/src/github.com/grpc-ecosystem/grpc-gateway/third_party/googleapis \
  --plugin=protoc-gen-grpc-java \
  --grpc-java_out=../../../java/client/src/main/java \
  --proto_path=. \
  model_center_service.proto

#generate python file
python3 -m grpc.tools.protoc -I. \
  -I/usr/local/include \
  -I$GOPATH/src/github.com/grpc-ecosystem/grpc-gateway/third_party/googleapis \
  --python_out=.. \
  message.proto

python3 -m grpc.tools.protoc -I. \
  -I/usr/local/include \
  -I$GOPATH/src/github.com/grpc-ecosystem/grpc-gateway/third_party/googleapis \
  --python_out=.. \
  --grpc_python_out=.. \
  metadata_service.proto

python3 -m grpc.tools.protoc -I. \
  -I/usr/local/include \
  -I$GOPATH/src/github.com/grpc-ecosystem/grpc-gateway/third_party/googleapis \
  --python_out=.. \
  --grpc_python_out=.. \
  model_center_service.proto

python3 -m grpc.tools.protoc -I. \
  -I/usr/local/include \
  -I$GOPATH/src/github.com/grpc-ecosystem/grpc-gateway/third_party/googleapis \
  --python_out=.. \
  --grpc_python_out=.. \
  deploy_service.proto

python3 -m grpc.tools.protoc -I. \
  -I/usr/local/include \
  -I$GOPATH/src/github.com/grpc-ecosystem/grpc-gateway/third_party/googleapis \
  --python_out=.. \
  --grpc_python_out=.. \
  scheduling_service.proto

python3 -m grpc.tools.protoc -I. \
  -I/usr/local/include \
  -I$GOPATH/src/github.com/grpc-ecosystem/grpc-gateway/third_party/googleapis \
  --python_out=.. \
  --grpc_python_out=.. \
  metric_service.proto

python3 -m grpc.tools.protoc -I. \
  -I/usr/local/include \
  -I$GOPATH/src/github.com/grpc-ecosystem/grpc-gateway/third_party/googleapis \
  --python_out=.. \
  --grpc_python_out=.. \
  high_availability.proto

cd ..

sed -i '' 's/\_ "github.com\/grpc-ecosystem\/grpc-gateway\/third_party\/googleapis\/google\/api"/\/\/\_ "github.com\/grpc-ecosystem\/grpc-gateway\/third_party\/googleapis\/google\/api"/g' go/ai_flow/metadata_service.pb.go
sed -i '' 's/\_ "github.com\/grpc-ecosystem\/grpc-gateway\/third_party\/googleapis\/google\/api"/\/\/\_ "github.com\/grpc-ecosystem\/grpc-gateway\/third_party\/googleapis\/google\/api"/g' go/ai_flow/model_center_service.pb.go
sed -i '' 's/\_ "github.com\/grpc-ecosystem\/grpc-gateway\/third_party\/googleapis\/google\/api"/\/\/\_ "github.com\/grpc-ecosystem\/grpc-gateway\/third_party\/googleapis\/google\/api"/g' go/ai_flow/deploy_service.pb.go
sed -i '' 's/\_ "github.com\/grpc-ecosystem\/grpc-gateway\/third_party\/googleapis\/google\/api"/\/\/\_ "github.com\/grpc-ecosystem\/grpc-gateway\/third_party\/googleapis\/google\/api"/g' go/ai_flow/scheduling_service.pb.go
sed -i '' 's/\_ "github.com\/grpc-ecosystem\/grpc-gateway\/third_party\/googleapis\/google\/api"/\/\/\_ "github.com\/grpc-ecosystem\/grpc-gateway\/third_party\/googleapis\/google\/api"/g' go/ai_flow/metric_service.pb.go
rm -rf go/*.go-E

sed -i -E 's/^import message_pb2 as message__pb2/from \. import message_pb2 as message__pb2/' *pb2*.py
sed -i -E 's/^import metadata_service_pb2 as metadata__service__pb2/from \. import metadata_service_pb2 as metadata__service__pb2/' *pb2*.py
sed -i -E 's/^import model_center_service_pb2 as model__center__service__pb2/from \. import model_center_service_pb2 as model__center__service__pb2/' *pb2*.py
sed -i -E 's/^import deploy_service_pb2 as deploy__service__pb2/from \. import deploy_service_pb2 as deploy__service__pb2/' *pb2*.py
sed -i -E 's/^import scheduling_service_pb2 as scheduling__service__pb2/from \. import scheduling_service_pb2 as scheduling__service__pb2/' *pb2*.py
sed -i -E 's/^import metric_service_pb2 as metric__service__pb2/from \. import metric_service_pb2 as metric__service__pb2/' *pb2*.py
sed -i -E 's/^import high_availability_pb2 as high__availability__pb2/from \. import high_availability_pb2 as high__availability__pb2/' *pb2*.py
rm -rf *.py-E
