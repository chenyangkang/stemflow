# Contributing to stemflow

We welcome contribution to `stemflow`

---

## Steps for contributing

1. Fork the git repository
1. Create a development environment
1. Create a new branch, make changes to code & add tests
1. Update the docs
1. Submit a pull request


We recommend you to open and issue of `feature` if you wish to add brand-new features before the PR, especially with heavy mathematics to resolve.

---

## Fork the git repository

You will need your own fork to work on the code. Go to the `stemflow` project page and hit the Fork button. You will want to clone your fork to your machine:

```bash
git clone git@github.com:YOUR-USER-NAME/stemflow.git stemflow-YOURNAME
cd stemflow-YOURNAME
git remote add upstream git://github.com/stemflow/stemflow.git
```

This creates the directory `stemflow-YOURNAME` and connects your repository to the upstream (main project) `stemflow` repository.

---


## Create a virtural development environment

Although `stemflow` has only a few dependency, we recommend creating a new environment to keep everything neat.

To do this, first install miniconda or conda:

- Install [miniconda](http://conda.pydata.org/miniconda.html) or [anaconda](http://docs.continuum.io/anaconda/)
- `cd` to the `stemflow` source directory


Next, start a new environment called `stemflow`, activate it, and install dependencies:

```bash
conda create -n stemflow -python=3.8 -y
activate stemflow
conda install --file requirement.txt
conda install pre-commit pytest pytest-cov pytest-xdist

pre-commit install
pip install -e .
```

This library uses `black`, `flake8` and `isort` pre-commit hooks. You should be familiar with [pre-commit](https://pre-commit.com/) before contributing.

---

## Create a new branch, edit code & add tests

Make changes to the code on a separate branch to keep you main branch clean:

```bash
git checkout -b shiny-new-feature
```

Make changes to your code and write tests as you go. Write clear, [self-documenting code](https://realpython.com/documenting-python-code/) to spend more time developing and less time describing how the code works.

If your branch is no longer up-to-date with `main`, run the following code to update it:

```bash
git fetch upstream
git rebase upstream/main
```

Testing is done with `pytest`, which you can run with either:

```bash
pytest -x -n auto --cov --no-cov-on-fail --cov-report=term-missing:skip-covered
```

---

## Update the docs

There are two places to update docs. One is required (docstrings), the other optional (`mkdocs` web documentation).

Adding docstrings to each new function/class is required. `stemflow` uses [Google-style docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) and, when you contribute to it, you should too. `mkdocs` automatically renders the API docs for all functions written with this style, so you don't need to re-document each function outside of the code.

If your code contributes important new features, or introduces novel/interesting concepts, write new documentation in the `docs/` directory. The docs system is managed by `mkdocs`, which renders from Markdown.

You can install `mkdocs` and the associated plugins with:

```bash
pip install mkdocs-material pillow cairosvg mkdocs-jupyter 'mkdocstrings[crystal,python]'
```

Then you can render the docs locally with:

```bash
mkdocs serve
```

## Commit

After you finish editing. Commit with words that summarize the changes.

```
git commit -m 'what I have changed'
```

You will possibly find that pre-commit trimmed your scripts. In this case you need to add those changed file again and commit again to save the changes.


---

## Submit a pull request

Once you’ve made changes and pushed them to your forked repository, you then submit a pull request to have them integrated into the `stemflow` code base.

For more information, you can find a PR tutorial in [GitHub’s Help Docs](https://help.github.com/articles/using-pull-requests/).

---
