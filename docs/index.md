# Welcome to Simulas suite!

```python
from matfact import matfact
from hmm_synthetic import hmm

for model in (matfact, hmm):
    model().fit(some_data)
```

