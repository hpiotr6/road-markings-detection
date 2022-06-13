# road-markings-detection

## Installation

```sh
# Install dependencies from pyproject.toml
poetry install
```
Alternatively
```sh
pip install -r requirements.txt   
```

## Poetry

### Setting up the environment

1. Install `poetry`: <https://python-poetry.org/docs/#installation>
2. Create an environment with `poetry install`
3. Run `poetry shell`
4. To add a new package run `poetry add <package>`. Don't forget to commit the lockfile.
5. To run unit tests for your service use `poetry run pytest` or simply `pytest` within `poetry shell`.
