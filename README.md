# FeatureSets.jl

[![CI](https://github.com/cursorinsight/FeatureSets.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/cursorinsight/FeatureSets.jl/actions/workflows/CI.yml)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

<!--[![codecov](https://codecov.io/gh/cursorinsight/FeatureSets.jl/branch/main/graph/badge.svg?token=P59CK9SA1Z)](https://codecov.io/gh/cursorinsight/FeatureSets.jl)-->

FeatureSets.jl is a Julia package designed around structures storing large
feature sets for classification and regression problems. It's designed to work
smoothly with [FeatureScreening.jl][], our solution to find significant features
in a high dimensional feature set, based on their importance in classification.

## Installation

FeatureSets.jl can be installed after adding Cursor Insight's [own
registry][CIJR] to the Julia environment:

```julia
julia> ]
pkg> registry add https://github.com/cursorinsight/julia-registry
     Cloning registry from "https://github.com/cursorinsight/julia-registry"
       Added registry `CursorInsightJuliaRegistry` to
       `~/.julia/registries/CursorInsightJuliaRegistry`

pkg> add FeatureSets
```

## `FeatureSet`

A `FeatureSet` object stores sample labels, feature names, and feature values
for each label/name combination. It also stores the date of creation and a
unique ID.

There are two constructor methods:

1. `FeatureSet(labels::AbstractVector{L},
   names::AbstractVector{N},
   features::AbstractMatrix{F})`
2. `FeatureSet(X::AbstractMatrix{F},
   y::AbstractVector{L})`

The first signature is the native API, expecting the sample labels and feature
names in vectors, and feature values in a matrix. The second signature uses the
*de facto* standard data science API, expecting an `X` feature matrix and a `y`
sample label vector. In this case, feature names are automatically assigned
integers from 1 going up. Both methods accept `id` and a `created_at` optional
keyword parameter to override the defaults.

```julia
# Stores feature names in a `names` field
julia> fs = FeatureSet([1, 2],       # labels
                       ["f1", "f2"], # names
                       [1 2;
                        3 4])        # features

# The `names` field contains the indices of the features.
julia> fs_without_feature_names = FeatureSet([1 2;
                                              3 4], # X
                                             [1, 2] # y)
```

Getters can be used to retrieve values from a `FeatureSet` object:

```julia
julia> id(fs) # return the unique ID of the feature set
julia> labels(fs) # return the label vector of the feature set
julia> names(fs) # return the name vector of the feature set
julia> features(fs) # return the feature matrix of the feature set
```

## HDF5 persistence

A `FeatureSet` can be written to and loaded from a [HDF5][] file:

```julia
julia> save("feature_sets/saved.hdf5", fs)
julia> save(fs; directory = "feature_sets") # file name generate from ID
julia> fs = load(FeatureSet, "feature_sets/$(fs_id).hdf5")
```

The `load` function accepts an optional `mmap` keyword argument. If that is set
to `true`, the feature matrix is memory mapped instead of fully loaded into the
memory, which can be useful (and significantly faster) for large feature sets.

The following HDF5 datasets are written to (and expected to be readable from)
the file:

* `created_at` (date-time formatted string): timestamp of the time of creation;
* `id` (string): a UUID of the feature set;
* `labels` (vector of *L* items): sample class labels;
* `names` (vector of *N* items): feature names;
* `features` (matrix of *L* rows and *N* columns): feature values.

[CIJR]: https://github.com/cursorinsight/julia-registry
[FeatureScreening.jl]: https://github.com/cursorinsight/FeatureScreening.jl
[HDF5]: https://www.hdfgroup.org/solutions/hdf5
