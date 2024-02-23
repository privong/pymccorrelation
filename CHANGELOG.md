# Changelog

## 0.2.x

### 0.2.6 (23 February 2024)

#### Bugfix

- Fixed a typo in Kendall's tau calculation for censored data that resulted in upper limit flags not being applied when evaluating whether Yi d< Yj (see Eq 27 of Isobe+ 1986, ApJ, 306, 490l). Identified by [Bennett-Skinner](https://github.com/Bennett-Skinner).

### 0.2.5 (28 September 2022)

#### Other

- `pymckendall()` is deprecated and will be removed after the 0.3.x series.
- Added a warning when `Nboot=1`. A user may provide this, intending to compute the correlation coefficient using many `Nperturb` rounds on the original dataset. However, `Nboot=1` does not use the input dataset as-is, but instead performs a single bootstrap with replacement. The warning clarifies that `Nboot=None` is the appropriate input when seeking to only use perturbation of points within the uncertainties.

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
