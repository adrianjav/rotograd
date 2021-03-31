# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.4] - 2021-03-31

### Changed

- RotoGrad no longer normalizes the losses by default (now it is only performed
  internally to compute GradNorm's weights). We keep the option to still normalize them 
  if desired.

[unreleased]: https://github.com/adrianjav/rotograd/compare/v0.1.4...HEAD
[0.1.4]: https://github.com/adrianjav/rotograd/compare/v0.1.3...v0.1.4
