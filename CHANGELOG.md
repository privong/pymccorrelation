# Changelog

## 0.2.x

### 0.2.3 (in preparation)

- fix description of `return_dist` behavior in doctrsing for `pymccorrelation`
- Amend changelog to fix release date for v0.2.2.
- Check x/y limit arrays to ensure they contain sensible values.

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
