# Purpose

Some ggml playground of mine. Never touched c++ or anything related to it in my life, might do dumb things here :)

Ggml library lets you run deep learning models efficiently and on a huge variety of devices, including phones and microcontrollers.

I want to try to build some models and get them running to see how well and fast it works. Learning experience FTW.

# Setting up

Clone the repo and get the submodules

```
git clone https://github.com/jezzarax/xornet.git
git submodule update --init --recursive
```

## Python part

- no venv or dependencies definition yet, might come soon. The usual deep learning stack is expected, pytorch, numpy, huggingface
- Also `python -m pip install ggml` is needed for model conversion

## C++ part

- `cmake` and `make` are needed to build the runner

# running

- get `train_xornet.py` and `conver_xornet.py` to run to the end
- build ggml model runner

```
mkdir build && cd build
cmake ..
make
./xornet ../models/xornet.gguf
```


# Models

## XorNet

Implementation of the classic XOR problem. 
The model is a simple 2 layer neural network with 2 input neurons, 2 hidden neurons and 1 output neuron. 
The model is trained using pytorch and then converted to ggml format.