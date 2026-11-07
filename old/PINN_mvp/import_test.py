import os
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

# Checking if all libraries work correctly


# Checking if pytorch works correctly
x = torch.rand(5, 3)
print(x)

# Checking if FeniCS works correctly

# Intentar importar FEniCS (silenciando warnings de JIT si es posible)

from fenics import *

