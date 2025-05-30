# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

## [v0.1.1](https://github.com/robbyzhaox/myocr/releases/tag/v0.1.1) - 2025-05-17

### Changed

- Fix error and optimize code for structured output pipeline

## [v0.1.0](https://github.com/robbyzhaox/myocr/releases/tag/v0.1.0) - 2025-05-14

### Changed

- Optimize the code to cope with situations of no text detection
- Polish documentation for release

## [v0.1.0-beta](https://github.com/robbyzhaox/myocr/releases/tag/v0.1.0-beta) - 2025-05-12

### Added

- Unify data structure of OCR result

### Changed

- Refactoring CommonOCRPipeline to use the new type OCRResult
- Polish code for CommonOCRPipeline & HTTP endpoint

## [v0.1.0-alpha.4](https://github.com/robbyzhaox/myocr/releases/tag/v0.1.0-alpha.4) - 2025-05-08

### Fixed
- fixed recognized text confidence

## [v0.1.0-alpha.3](https://github.com/robbyzhaox/myocr/releases/tag/v0.1.0-alpha.3) - 2025-05-07

### Fixed
- fixed workflow for building docker image

## [v0.1.0-alpha.2](https://github.com/robbyzhaox/myocr/releases/tag/v0.1.0-alpha.2) - 2025-05-07

### Added

- add workflow for releasing docker image
- update readme
- add config for release doc manually

### Fixed
- fixed char decode for black space

## [v0.1.0-alpha.1](https://github.com/robbyzhaox/myocr/releases/tag/v0.1.0-alpha.1) - 2025-05-06

### Added
- version check in releash.sh
- add demo url to readme

### Fixed

- fix logging issue by moving logging config out from myocr package

## [v0.1.0-alpha](https://github.com/robbyzhaox/myocr/releases/tag/v0.1.0-alpha) - 2025-05-04

