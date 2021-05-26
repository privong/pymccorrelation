# Changelog

## 0.2.x

### 0.2.4 (26 May 2021)

#### Bugfixes

- When computing Kendall's tau with censored data and bootstrapping, the x/y limit flags were not resampled. This is fixed and the limit flags are now properly resampled.

### 0.2.3 (25 August 2020)

#### Enhancements

- Check x/y limit arrays to ensure they contain sensible values.
- Update assertions on array lengths to provide more helpful debugging information.

#### Bugfixes

- fix description of `return_dist` behavior in doctrsing for `pymccorrelation`
- Amend changelog to fix release date for v0.2.2.
- Fix missing pass of x/y limits through `pymckendall()` wrapper

### 0.2.2 (2020 July 08)

#### Bugfixes

- Pearson's r was incorrectly imported as spearman's rho. The import has been fixed.

### 0.2.1 (2020 July 01)

- Add `setup.py` and package for pypi release.

### 0.2.0 (2020 June 22)

#### Enhancements

- Rewrite random number generation to use numpy's newer Generator approach.

#### Bugfixes

- Fix bug in bootstrapping that would cause final entry in list to be skipped.

## 0.1.x

### 0.1.0 (2020 June 18)

Initial Release
