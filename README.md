# MixtureLib

## Requirements
1. Python 3.6.2
2. pip 19.2.3

## Installation
1. It is recommended that you create a virtual environment.
```
python3.6 -m venv grpcExmplEnv
```
2. Switch to virtual environment (all python packages will be installed there).
```
source grpcExmplEnv/bin/activate
```
3. Download this repository.
```
git clone https://github.com/andriygav/grpcExmpl.git
cd grpcExmpl
```
4. Install required packages.
```
python3.6 -m pip install -r requirements.txt
```
5. Generate new python grpc protocol files. This step is required only if the example.proto file is changed.
```
python3.6 -m grpc_tools.protoc \
    -I ./src/example_protos/proto/ \
    --python_out=./src/example_protos/ \
    --grpc_python_out=./src/example_protos/ \
    ./src/example_protos/proto/example_protos/*.proto
```
6. Install service and protos.
```
python3.6 -m pip install src/example_protos/.
python3.6 -m pip install src/example/.
```

## Running
1. Run service. All service settings in config/example.cfg file.
```
example_service config/example.cfg
```
2. There are two ways to use the service:
  * by client (need to be written)
  * by grpcui (need to be installed)
3.1. By client.
```
example_client [-h] [--type {sum,prod}] [--server SERVER] a b
```
3.2. By grpcui. (default service port is 9878). After executing the following command, go to the browser and write “gRPC Web UI” in the URL bar.
```
grpcui -plaintext [::]:9878
```
## Version
0.0.1
