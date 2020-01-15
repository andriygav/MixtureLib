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
6. Install service.
```
python3.6 -m pip install src/example/.
```


## Version
0.0.1
