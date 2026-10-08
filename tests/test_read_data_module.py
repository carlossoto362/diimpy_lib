
from diimpy.read_data_module import read_constants

from diimpy.settings import DEFAULT_SETTINGS_DIR
from pathlib import Path
from typing import Union
from os import PathLike
import torch

if __name__ == "__main__":
    
    constants = read_constants(lambdas_path=DEFAULT_SETTINGS_DIR /  'lambdas',lambdas_column_name='lambdas',cte_path=DEFAULT_SETTINGS_DIR / 'cte.csv',tensor = True,my_device = 'cpu',dict=False,precision=torch.float32)

    print(constants.scattering_w)
