# omar

[![PyPI version](https://badge.fury.io/py/omar.svg)](https://badge.fury.io/py/omar) WIP (name conflict see https://github.com/pypi/support/issues/6159)

`omar` (**O**pen **M**ultivariate **A**daptive **R**egression) is a Python package for discovering localised, linear structure in complex, high-dimensional datasets. It implements a modernized version of the Multivariate Adaptive Regression Splines (MARS) algorithm with improved numerical efficiency, based on modern rank-one update strategies and optional Fortran acceleration.

## Installation

### From PyPI *(coming soon)*

```bash
pip install omar
```

### From Source

See the [Getting Started](https://github.com/Helge-Stein-Group/omar/wiki/Getting-Started) wiki page.

---

## Quick Example

```python
import numpy as np
from omar import OMAR

np.random.seed(0)

x = np.random.normal(2, 1, size=(1000, 3))
noise = np.random.normal(size=1000)
y = ((x[:, 0] + np.maximum(0, (x[:, 0] - 1)) +
      np.maximum(0, (x[:, 0] - 1)) * x[:, 1] +
      np.maximum(0, (x[:, 0] - 1)) * np.maximum(0, (x[:, 1] - 0.8))) +
     0.12 * noise)
model = OMAR(max_nbases=11)
model.find_bases(X, y)
print(model)
```
## Documentation

Visit [omar wiki](https://github.com/Helge-Stein-Group/omar/wiki)

---

## Contributing

We welcome contributions! See [Contributing](https://github.com/Helge-Stein-Group/omar/wiki/Contributing) for code guidelines, tests, and CI workflow.

---



## License

This project is licensed under the MIT License. See `LICENSE.txt`.
