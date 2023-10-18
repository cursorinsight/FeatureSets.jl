###-----------------------------------------------------------------------------
### Copyright (C) The FeatureSets.jl contributors
###
### SPDX-License-Identifier: MIT
###-----------------------------------------------------------------------------

module DecisionTreeExt

###=============================================================================
### Imports
###=============================================================================

using DecisionTree: Ensemble as RandomForest, Leaf, Node
using FeatureSets: AbstractFeatureSet, features, labels, names

import DecisionTree: build_forest, nfoldCV_forest
import DecisionTree: apply_forest, confusion_matrix

###=============================================================================
### Implementation
###=============================================================================

"""
    build_forest(feature_set::AbstractFeatureSet; config::NamedTuple)

Given a `feature_set`, build a random forest from its [`features`](@ref), using
its [`labels`](@ref) as the target variable. The forest configuration is
specified by `config`.
"""
function build_forest(feature_set::AbstractFeatureSet; config = (;), kwargs...)
    return __build_forest(labels(feature_set),
                          features(feature_set);
                          config,
                          kwargs...)
end

"""
    nfoldCV_forest(feature_set::AbstractFeatureSet; config::NamedTuple)

Given a `feature_set`, perform an n-fold verification with random forests from
its [`features`](@ref), using its [`labels`](@ref) as the target variable. The
forest configuration is specified by `config`.
"""
function nfoldCV_forest(feature_set::AbstractFeatureSet;
                        config = (;),
                        verbose = false)
    return __nfoldCV_forest(labels(feature_set),
                            features(feature_set);
                            config,
                            verbose)
end

"""
    apply_forest(forest::RandomForest,
                 feature_set::AbstractFeatureSet;
                 use_multithreading::Bool = false)

Apply a random `forest` on the [`features`](@ref) of a `feature_set`. Return the
vector of predicted labels.
"""
function apply_forest(forest::RandomForest,
                      feature_set::AbstractFeatureSet;
                      use_multithreading::Bool = false)
    return apply_forest(forest, features(feature_set); use_multithreading)
end

"""
    confusion_matrix(forest::RandomForest,
                     feature_set::AbstractFeatureSet;
                     use_multithreading::Bool = false)

Apply a random `forest` on the [`features`](@ref) of a `feature_set`, and build
a confusion matrix of the predictions and the ground truth [`labels`](@ref).
"""
function confusion_matrix(forest::RandomForest,
                          feature_set::AbstractFeatureSet;
                          use_multithreading::Bool = false)
    predicted = apply_forest(forest, feature_set; use_multithreading)
    return confusion_matrix(labels(feature_set), predicted)
end

###-----------------------------------------------------------------------------
### `DecisionTree` wrappers
###-----------------------------------------------------------------------------

const DEFAULT_BUILD_FOREST_CONFIG =
    (n_subfeatures          = -1,
     n_trees                = 10,
     partial_sampling       = 0.7,
     max_depth              = -1,
     min_samples_leaf       = 1,
     min_samples_split      = 2,
     min_purity_increase    = 0.0)

function __build_forest(labels::AbstractVector{L},
                        features::AbstractMatrix{F};
                        config::NamedTuple = (;),
                        kwargs...
                       )::RandomForest{F, L} where {L, F}
    config::NamedTuple = (; DEFAULT_BUILD_FOREST_CONFIG..., config...)
    return build_forest(labels,
                        features,
                        config.n_subfeatures,
                        config.n_trees,
                        config.partial_sampling,
                        config.max_depth,
                        config.min_samples_leaf,
                        config.min_samples_split,
                        config.min_purity_increase;
                        kwargs...)
end

const DEFAULT_NFOLDCV_FOREST_CONFIG =
    (n_folds                = 4,
     n_subfeatures          = -1,
     n_trees                = 10,
     partial_sampling       = 0.7,
     max_depth              = -1,
     min_samples_leaf       = 1,
     min_samples_split      = 2,
     min_purity_increase    = 0.0)

function __nfoldCV_forest(labels::AbstractVector,
                          features::AbstractMatrix;
                          config::NamedTuple = (;),
                          kwargs...)
    config::NamedTuple = (; DEFAULT_NFOLDCV_FOREST_CONFIG..., config...)
    return nfoldCV_forest(labels,
                          features,
                          config.n_folds,
                          config.n_subfeatures,
                          config.n_trees,
                          config.partial_sampling,
                          config.max_depth,
                          config.min_samples_leaf,
                          config.min_samples_split,
                          config.min_purity_increase;
                          kwargs...)
end

end # module DecisionTreeExt
