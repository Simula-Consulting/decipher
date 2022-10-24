# decipher
The code developed for the DeCipher project

# Installation

## Development
We use [Mookme](mookme.org) as Git hook manager. Additionally, scripts from `pre-commit-hooks` are used.
To install and configure, run
```bash
pip install --user pre-commit-hooks  # Install to user. Alternatively use venv.
npm install @escape.tech/mookme  # Install Mookme to directory
npx mookme init --only-hook  # Set up hooks in repo
```
from the root of the repo.
This requires npm.

The hooks are run on each commit.
You may also run the hooks manually at any time with
```bash
npx mookme run -t pre-commit --all
```
