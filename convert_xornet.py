import gguf

import torch
import torch.nn as nn
from safetensors.torch import load_model
class XORNet(nn.Module):
    def __init__(self):
        super(XORNet, self).__init__()
        self.fc1 = nn.Linear(2, 2) # Input to hidden layer
        self.fc2 = nn.Linear(2, 1) # Hidden to output layer

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

model_name = "xornet"
load_model(m, f"./models/{model_name}.safetensors")

gguf_model_file = f"./models/{model_name}.gguf"
gguf_writer = gguf.GGUFWriter(gguf_model_file, model_name)

for k, v in m.state_dict().items():
    gguf_writer.add_tensor(k, v.numpy())

gguf_writer.write_header_to_file()
gguf_writer.write_kv_data_to_file()
gguf_writer.write_tensors_to_file()
gguf_writer.close()
print("Model converted and saved to '{}'".format(gguf_model_file))