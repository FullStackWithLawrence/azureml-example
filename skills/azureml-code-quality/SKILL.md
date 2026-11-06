---
name: azureml-code-quality
description: Use before finishing any azureml-example change, to lint and format Python, Markdown, JSON and YAML with the repository's pre-commit hooks (pyupgrade, codespell, Prettier, black, flake8, isort, bandit, pre-commit-hooks, commitlint), and when writing docstrings or log calls. Explains what each hook enforces and doesn't, pylint's role, how to run the hooks on changed files under zsh, and the style conventions of the code.
---

# Code Quality

`.pre-commit-config.yaml` is the standard. If a hook passes, the code meets
it; if it fails, fix the code. Never bypass a hook (`--no-verify`, `SKIP=`, a
blanket `# noqa`).

## What the hooks enforce

| Hook             | Configuration                                  | Notes                                                                                                                          |
| ---------------- | ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| pyupgrade        | default                                        | Runs under the default Python 3.14 (no `--py3xx-plus` flag)                                                                    |
| codespell        | `.codespellrc`, `--ignore-words=codespell.txt` | Also checks Markdown and notebooks (except the one skipped in `.codespellrc`). Add real project words to `codespell.txt`       |
| prettier         | `.prettierrc.json`, `.prettierignore`          | Markdown, JSON, YAML. Let it reformat                                                                                          |
| black            | `pyproject.toml`, line length 120              | Formatter                                                                                                                      |
| flake8           | `.flake8`                                      | `select = C101` only (coding magic comment). Not a style check                                                                 |
| isort            | `pyproject.toml`: `profile = black`            | Import order; black then sets the blank lines after imports                                                                    |
| bandit           | `-ll`                                          | Medium and high findings fail. `pickle.load` carries `# nosec B301` with a reason                                              |
| pre-commit-hooks | v5                                             | JSON/TOML/XML/YAML syntax, merge markers, symlinks, private keys, AWS keys, debug statements, trailing whitespace, end-of-file |
| commitlint       | `commitlint.config.js`, commit-msg stage       | See `azureml-git-commits`                                                                                                      |

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
  though they need a `python3.14` on the PATH for the Python hooks.

## Keep it simple

The user's rule of thumb: **one operation per line.** It isn't absolute, but
when there's a choice between compact and simple, choose simple. Students read
this code line by line, and each line should say one thing.

- Give an intermediate result a name instead of nesting calls or chaining
  `or` fallbacks across several sources:

  ```python
  # no
  best = (job.tags or {}).get("automl_best_child_run_id") or (job.properties or {}).get("automl_best_child_run_id")

  # yes
  tags = job.tags or {}
  best = tags.get("automl_best_child_run_id")
  ```

- Prefer a plain `for` loop with an `if` over a comprehension that filters,
  transforms and calls methods at once. A simple one-step comprehension
  (`[c.name for c in clusters]`) is fine.
- No conditional expressions inside arguments, f-strings or other
  expressions; set the value on its own line first.
- One call per line where the result of one call feeds the next
  (`session = load_model()`, then `outputs = session.run(...)`), not
  `load_model().run(...)[0]`.
- A function does one job. Split out a helper when a function needs a
  comment to say where one part ends and the next begins.

## Style

- Match the surrounding code: names, comment density, `logger =
getLogger(__name__)`, %-style lazy log arguments, re-raising after logging.
- Google-style docstrings: a summary line, then `Args:`, `Returns:`,
  `Raises:`. Every public class, method and function has one, and it matches
  the signature.
- Module docstrings of source and command files start with the author block
  (`Lawrence McDaniel` / `https://lawrencemcdaniel.com`) and, for commands, a
  `usage:` line.
- Modern type hints (`str | None`, `dict`, `list`), not `typing.Optional`.
  pyright (what Pylance shows in the editor) and
  `mypy azure_ai/ml_studio.py azure_ai/commands` should report no errors; they
  aren't hooks, but the user sees Pylance's errors. Narrow `X | None` with a check that raises, rather
  than `# type: ignore`.
- pydocstringformatter makes a docstring's first line a sentence ending in a
  period. If that line wraps mid-sentence it inserts a period and a blank line
  in the middle of the sentence: write a short, complete first line, then
  check what the hook did.
- Comments explain Azure behavior a student wouldn't guess (why an MLTable
  file, why `delete()` archives), not what the next line does.
- Don't reformat code you didn't change.
