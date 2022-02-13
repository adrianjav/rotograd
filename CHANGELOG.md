# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.5.1] - 2022-02-13

### Added

- Working toy example (folder `example`).

### Changed

- Fixed small bugs in RotoGrad (attribute names and properties inheritance).

## [0.1.5] - 2021-12-17

### Added

- Introduced RotateOnly class. This is equivalent to the RotoGrad class of version 0.1.4.
- Added `backbone_loss` argument to backward to allow for backbone-specific losses.

### Changed

- Updated VanillaMTL to have the same methods as RotoGrad class.
- Modified RotoGrad so that it also scales the gradients as described in the updated version of the paper.
- Enabled gradient computation when computing the normalizing factor of RotoGradNorm.
  
## [0.1.4] - 2021-03-31

### Changed

- RotoGrad no longer normalizes the losses by default (now it is only performed
  internally to compute GradNorm's weights). We keep the option to still normalize them 
  if desired.

[unreleased]: https://github.com/adrianjav/rotograd/compare/v0.1.4...HEAD
[0.1.4]: https://github.com/adrianjav/rotograd/compare/v0.1.3...v0.1.4
[0.1.5]: https://github.com/adrianjav/rotograd/compare/v0.1.4...v0.1.5
[0.1.5.1]: https://github.com/adrianjav/rotograd/compare/v0.1.5...v0.1.5.1
