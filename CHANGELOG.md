# Change Log

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/) and this project adheres to [Semantic Versioning](http://semver.org/).



## [2.0.1](https://github.com/FullStackWithLawrence/azureml-example/compare/v2.0.0...v2.0.1) (2026-11-06)

### Bug Fixes

* make the dataset commands and automl jobs work against azure ([4a8713f](https://github.com/FullStackWithLawrence/azureml-example/commit/4a8713f2d6ff983e67ac036af18dc617adcf0442))

## [2.0.0](https://github.com/FullStackWithLawrence/azureml-example/compare/v1.0.0...v2.0.0) (2026-11-06)

### ⚠ BREAKING CHANGES

* requires Python 3.14 and the Azure ML SDK v2. Rebuild the virtual environment with make init. Code that used the v1 objects these classes returned (Dataset, ComputeTarget, Run) must use their v2 equivalents (Data, AmlCompute, Job). get_default_name() takes an MLClient. get_best_model() returns the best trial job instead of a (run, model) pair. A dataset's delete() archives it. Re-register data assets created by the old code, so that they are MLTable.

Co-authored-by: Lawrence McDaniel <lpm0073@gmail.com>

### Features

* **function:** serve the automl model as onnx on python 3.14 ([4c7447b](https://github.com/FullStackWithLawrence/azureml-example/commit/4c7447b9fe9864849fdaac267e188cb3ed90403f))
* move to python 3.14 and the azure machine learning sdk v2 ([a92d567](https://github.com/FullStackWithLawrence/azureml-example/commit/a92d567387bd3052441630847909b74525c5830d))

## 1.0.0 (2025-07-03)

### Features

- add basic set of commands ([133f9e2](https://github.com/FullStackWithLawrence/azureml-example/commit/133f9e2833c937ba7e5225eead3bb73e846e070d))
