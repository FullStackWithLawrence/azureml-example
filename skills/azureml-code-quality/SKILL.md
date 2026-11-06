---
name: azureml-code-quality
description: Use before finishing any azureml-example change, to lint and format Python, Markdown, JSON and YAML with the repository's pre-commit hooks (pyupgrade, codespell, Prettier, black, flake8, isort, bandit, pre-commit-hooks, commitlint), and when writing docstrings or log calls. Explains what each hook enforces and doesn't, pylint's role, how to run the hooks on changed files under zsh, and the style conventions of the code.
---

# Code Quality

`.pre-commit-config.yaml` is the standard. If a hook passes, the code meets
it; if it fails, fix the code. Never bypass a hook (`--no-verify`, `SKIP=`, a
blanket `# noqa`).

## What the hooks enforce

| Hook             | Configuration                                                  | Notes                                                                                                                          |
| ---------------- | -------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| pyupgrade        | default                                                        | Runs under the default Python 3.9                                                                                              |
| codespell        | `.codespellrc`, `--ignore-words=codespell.txt`                 | Also checks Markdown and notebooks (except the one skipped in `.codespellrc`). Add real project words to `codespell.txt`       |
| prettier         | `.prettierrc.json`, `.prettierignore`                          | Markdown, JSON, YAML. Let it reformat                                                                                          |
| black            | `pyproject.toml`, line length 120                              | Formatter                                                                                                                      |
| flake8           | `.flake8`                                                      | `select = C101` only (coding magic comment). Not a style check                                                                 |
| isort            | `pyproject.toml`: `profile = black`, `lines_after_imports = 2` | Two blank lines after imports                                                                                                  |
| bandit           | `-ll`                                                          | Medium and high findings fail. `pickle.load` carries `# nosec B301` with a reason                                              |
| pre-commit-hooks | v5                                                             | JSON/TOML/XML/YAML syntax, merge markers, symlinks, private keys, AWS keys, debug statements, trailing whitespace, end-of-file |
| commitlint       | `commitlint.config.js`, commit-msg stage                       | See `azureml-git-commits`                                                                                                      |

The `eslint` hook targets a `client/` folder that doesn't exist; it never runs.

**pylint** isn't a hook (it's commented out) but `make lint` runs it with
`.pylintrc`. Existing code uses targeted disables such as
`# pylint: disable=W0718` above a deliberate broad `except`. Keep new code free
of new pylint warnings where that's easy; don't sweep old ones.

## How to run

```bash
source venv/bin/activate
git diff --name-only | xargs pre-commit run --files
git ls-files --others --exclude-standard | xargs pre-commit run --files   # new files
```

- Use `xargs`: under zsh `pre-commit run --files $FILES` passes one argument
  and checks nothing.
- Hooks that rewrite files (black, isort, Prettier, end-of-file-fixer) fail
  the first run and pass the second. Run twice, and read the diff.
- `--all-files` skips untracked files.
- `make pre-commit` runs `pre-commit autoupdate`, which edits
  `.pre-commit-config.yaml`. Don't run it as part of another change.
- If the project venv is broken (`azureml-python-environment`), any
  `pre-commit` on the PATH works: the hooks bring their own environments,
  though they need a `python3.9` on the PATH for the Python hooks.

## Style

- Match the surrounding code: names, comment density, `logger =
getLogger(__name__)`, %-style lazy log arguments, re-raising after logging.
- Google-style docstrings: a summary line, then `Args:`, `Returns:`,
  `Raises:`. Every public class, method and function has one, and it matches
  the signature.
- Module docstrings of source and command files start with the author block
  (`Lawrence McDaniel` / `https://lawrencemcdaniel.com`) and, for commands, a
  `usage:` line.
- Type hints from `typing` (`Optional`, `Union`) for 3.9 compatibility.
- Comments explain Azure behavior a student wouldn't guess (why
  `validate=False`, why a clean upload directory), not what the next line does.
- Don't reformat code you didn't change.
