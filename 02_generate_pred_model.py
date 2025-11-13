from predify import predify
from utility import get_model
from config import model_name, version


config_path = f"config_pnet/config_P{model_name.upper()}_{version}.toml"
output_address = f"pnet/p{model_name.lower()}{version}.py"

model = get_model(pretrained=False, ngpus=0, model=model_name)

predify(model, config_path, output_address=output_address)
