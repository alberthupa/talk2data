Short answer: use `code_path` (and optionally `loader_module`) to bundle multiple files/folders, and `artifacts` for non-code files.

### Typical pattern (multi-file project)

```
project/
  my_agent/
    __init__.py
    model.py        # defines class MyModel(mlflow.pyfunc.PythonModel)
    utils.py
  conf/
    prompts/
    config.yaml
  log_model.py
```

**log_model.py**

```python
import mlflow, mlflow.pyfunc as pf
from my_agent.model import MyModel

mlflow.set_experiment("/Shared/agents")
with mlflow.start_run():
    pf.log_model(
        artifact_path="agent",
        python_model=MyModel(),
        code_path=["my_agent"],                      # one or more dirs/files
        artifacts={"config":"conf/config.yaml", "prompts":"conf/prompts"},  # data dirs/files
        pip_requirements=["-r requirements.txt"]     # or conda_env=...
    )
```

Notes:

* `code_path` accepts a list of local dirs/files; MLflow zips them and adds to `PYTHONPATH` when loading/serving the model.
* Put all your Python modules under a package (e.g., `my_agent/`) and include that directory in `code_path`. You can pass multiple paths if needed.
* Use `artifacts` for non-code assets (YAMLs, prompt folders, models, etc.). Values can be files or directories; they’ll be available under `context.artifacts[...]` in your `PythonModel`.
* Prefer `pip_requirements`/`extra_pip_requirements` (or `conda_env`) to lock runtime deps.

### Alternative: import at load time

If you don’t want to import `MyModel()` at logging time, you can ship code and specify a loader:

```python
pf.log_model(
    artifact_path="agent",
    loader_module="my_agent.model",   # module must implement _load_pyfunc(model_path)
    code_path=["my_agent"],
    artifacts={"config":"conf/config.yaml"}
)
```

Implement `_load_pyfunc(model_path)` in `my_agent/model.py` to return a `PythonModel`.

### Packaging as a wheel (optional)

Build your code as a wheel, upload it to DBFS or a package repo, then in `pip_requirements` include that wheel path (or versioned package). This keeps `code_path` minimal and moves versioning to packaging.

That’s it—`code_path` for multi-file Python, `artifacts` for data, env pinned via `pip_requirements`/`conda_env`.
