# MatFact

Matrix Factorization with temporal regularization.

!!! example
    ```python
    from matfact.model import train_and_log
    from matfact.data_generation import Dataset
    
    data = Dataset.generate()
    X_train, X_test, _, _ = data.get_split_X_M()
    
    train_and_log(X_train, X_test)
    ```

::: matfact.model.matfact
    options:
      show_root_heading: true

::: matfact.model.config
    options:
      members: [ModelConfig]
      show_root_heading: true

::: matfact.model
    options:
      members: 
        - train_and_log
      show_root_heading: true
      separate_signature: true

::: matfact.model.factorization
    options:
      show_root_heading: true
      filters:
        - "!loss()$"
        - "!run_step()"
        - "!^_"

::: matfact.model.factorization.convergence

::: matfact.model.logging
    options:
      show_root_heading: true
      filters:
        - "!MLFlowRunHierarchyException"
        - "!^_"

::: matfact.data_generation.Dataset
    options:
      show_root_heading: true

::: matfact.model.predict.dataset_utils

::: matfact.model.predict.classification_tree