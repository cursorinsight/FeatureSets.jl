###-----------------------------------------------------------------------------
### Copyright (C) The FeatureSets.jl contributors
###
### SPDX-License-Identifier: MIT
###-----------------------------------------------------------------------------

"""
# module FeatureSets

FeatureSets.jl is a Julia package designed around structures storing large
feature sets for classification and regression problems. It's designed to work
smoothly with [FeatureScreening.jl][], our solution to find significant features
in a high dimensional feature set, based on their importance in classification.

# struct FeatureSet

A [`FeatureSet`](@ref) object stores sample labels, feature names, and feature
values for each label/name combination. It also stores the date of creation and
a unique ID.

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
keyword parameter to override the defaults."

Getters can be used to retrieve values from a `FeatureSet` object:

```
julia> id(fs) # return the unique ID of the feature set
julia> labels(fs) # return the label vector of the feature set
julia> names(fs) # return the name vector of the feature set
julia> features(fs) # return the feature matrix of the feature set
```
"""
module FeatureSets

###=============================================================================
### Exports
###=============================================================================

# Types and accessors
export AbstractFeatureSet, FeatureSet
export features, labels, names

# I/O
export load, save

###=============================================================================
### Includes
###=============================================================================

include("Utilities.jl")

###=============================================================================
### Imports
###=============================================================================

### Struct
using Base: @kwdef
using Dates: DateTime, UTC, now
using UUIDs: UUID, uuid4

### Getters
import Base: names

### Base API
import Base: ==, getindex, hash, parent, show, view

## Size API
import Base: axes, length, ndims, size

## Iterable API
import Base: eachcol, eachrow, iterate

### File API
using HDF5: File as HDF5File
using HDF5: create_dataset, dataspace, datatype, h5open, readmmap

### Others
import Base: merge, rand

### Internals
using PackageExtensionCompat: @require_extensions
using .Utilities: @unimplemented, to_hdf5

###=============================================================================
### AbstractFeatureSet
###=============================================================================

"""
    AbstractFeatureSet{L, N, F}

Abstract base type for storing feature values by labels and feature names.

# Type parameters:

- `L`: Type of the labels.
- `N`: Type of the feature names.
- `F`: Type of the feature values, typically some numeric type.

# Example

|           | "feature 1" | "feature 2" |   ...   | "feature N" |
|:---------:|:-----------:|:-----------:|:-------:|:-----------:|
| "label-1" |   101.001   |   431.331   |   ...   |   20.9221   |
| "label-1" |   121.340   |   421.393   |   ...   |   21.3419   |
|    ...    |     ...     |     ...     |   ...   |     ...     |
| "label-M" |   131.349   |   134.119   |   ...   |   -0.1124   |
| "label-M" |   128.218   |   329.218   |   ...   |   10.0038   |
"""
abstract type AbstractFeatureSet{L, N, F} end

##------------------------------------------------------------------------------
## Abstract API
##------------------------------------------------------------------------------

@unimplemented function labels(::AbstractFeatureSet) end
@unimplemented function names(::AbstractFeatureSet) end
@unimplemented function features(::AbstractFeatureSet) end
@unimplemented function merge(::AbstractFeatureSet, ::AbstractFeatureSet) end
@unimplemented function getindex(::AbstractFeatureSet, inds...) end
@unimplemented function view(::AbstractFeatureSet, inds...) end
parent(fs::AbstractFeatureSet) = fs

##------------------------------------------------------------------------------
## Base API
##------------------------------------------------------------------------------

function show(io::IO, features::T)::Nothing where {T <: AbstractFeatureSet}
    (height, width) = size(features)
    print(io,
          "$(T)<$(height) × $(width)>",
          parent(features) !== features ? " view" : "")
    return nothing
end

function ==(a::AbstractFeatureSet, b::AbstractFeatureSet)
    return typeof(a) == typeof(b) &&
        labels(a) == labels(b) &&
        names(a) == names(b) &&
        features(a) == features(b)
end

function hash(feature_set::AbstractFeatureSet, h::UInt64)::UInt64
    parts = [labels(feature_set), names(feature_set), features(feature_set)]
    return reduce(parts; init = h) do h, part
        return hash(part, h)
    end
end

##------------------------------------------------------------------------------
## Size API
##------------------------------------------------------------------------------

function ndims(feature_set::AbstractFeatureSet)::Int
    return ndims(features(feature_set))
end

function axes(feature_set::AbstractFeatureSet)::Tuple
    return ntuple(i -> axes(feature_set, i), ndims(feature_set))
end

function size(feature_set::AbstractFeatureSet)::Tuple{Int, Int}
    return size(features(feature_set))
end

function size(feature_set::AbstractFeatureSet, dim::Int)::Int
    return size(features(feature_set), dim)
end

function length(feature_set::AbstractFeatureSet)::Int
    return size(features(feature_set), 1)
end

function merge(xs::AbstractFeatureSet...)
    return reduce(merge, xs)
end

##------------------------------------------------------------------------------
## Iterable API
##------------------------------------------------------------------------------

function iterate(feature_set::AbstractFeatureSet)
    return iterate(eachrow(feature_set))
end

function iterate(feature_set::AbstractFeatureSet, state)
    return iterate(eachrow(feature_set), state)
end

function eachrow(feature_set::AbstractFeatureSet)
    return zip(labels(feature_set), eachrow(features(feature_set)))
end

function eachcol(feature_set::AbstractFeatureSet)
    return zip(names(feature_set), eachcol(features(feature_set)))
end

###=============================================================================
### FeatureSet
###=============================================================================

"""
    FeatureSet{L, N, F}

Reference implementation of abstract base type `AbstractFeatureSet{L, N, F}`,
for storing feature values by labels and feature names in a matrix.
"""
@kwdef struct FeatureSet{L, N, F} <: AbstractFeatureSet{L, N, F}
    id::UUID = uuid4()
    created_at::DateTime = now(UTC)

    labels::AbstractVector{L}
    names::AbstractVector{N}
    features::AbstractMatrix{F}

    __name_indices::Dict{N, Int}
    __parent::Union{FeatureSet, Nothing} = nothing
end

"""
    FeatureSet(labels::AbstractVector{L},
               names::AbstractVector{N},
               features::AbstractMatrix{F};
               id::UUID = uuid4(),
               created_at::DateTime = now(UTC)
              )::FeatureSet{L, N, F} where {L, N, F}
"""
function FeatureSet(labels::AbstractVector{L},
                    names::AbstractVector{N},
                    features::AbstractMatrix{F};
                    kwargs...
                   )::FeatureSet{L, N, F} where {L, N, F}
    @assert (length(labels), length(names)) == size(features)

    __name_indices::Dict{N, Int} = names |> enumerate .|> reverse |> Dict

    return FeatureSet{L, N, F}(;
                               labels,
                               names,
                               features,
                               __name_indices,
                               kwargs...)
end

"""
    FeatureSet(X, y)

Create a `FeatureSet` from a feature matrix and vector of labels.

Classic data science API.
"""
function FeatureSet(X::AbstractMatrix{F},
                    y::AbstractVector{L};
                    kwargs...
                   )::FeatureSet{L, Int, F} where {L, F}
    return FeatureSet(y, 1:size(X, 2), X; kwargs...)
end

###-----------------------------------------------------------------------------
### Getters
###-----------------------------------------------------------------------------

function id(feature_set::FeatureSet)::UUID
    return feature_set.id
end

function created_at(feature_set::FeatureSet)::DateTime
    return feature_set.created_at
end

function labels(feature_set::FeatureSet{L}
               )::AbstractVector{L} where {L}
    return feature_set.labels
end

function names(feature_set::FeatureSet{L, N}
              )::AbstractVector{N} where {L, N}
    return feature_set.names
end

function features(feature_set::FeatureSet{L, N, F}
                 )::AbstractMatrix{F} where {L, N, F}
    return feature_set.features
end

###-----------------------------------------------------------------------------
### Base API
###-----------------------------------------------------------------------------

function axes(feature_set::FeatureSet, dim::Int)::AbstractVector
    return dim == 2 ? names(feature_set) : axes(features(feature_set), dim)
end

IndexType{T} = Union{<: T, AbstractVector{<: T}, Colon}

for lookup in [:getindex, :view]
    kwargs = lookup == :view ? [:(:__parent => feature_set)] : []
    @eval function $lookup(feature_set::FeatureSet{L, N},
                           label_index::IndexType{Integer},
                           name_index::IndexType{N}) where {L, N}
        rows = label_index
        cols = resolve_name_index(feature_set, name_index)

        if rows isa Integer || cols isa Integer
            return $lookup(features(feature_set), rows, cols)
        end

        return FeatureSet($lookup(labels(feature_set), rows),
                          $lookup(names(feature_set), cols),
                          $lookup(features(feature_set), rows, cols);
                          $(kwargs...))
    end
end

function resolve_name_index(feature_set::FeatureSet, ::Colon)::Colon
    return (:)
end
function resolve_name_index(feature_set::FeatureSet{_L, N},
                            name::N
                           )::Int where {_L, N}
    return feature_set.__name_indices[name]
end

function resolve_name_index(feature_set::FeatureSet{_L, N},
                            names::AbstractVector{<: N}
                           )::Vector{Int} where {_L, N}
    return resolve_name_index.(Ref(feature_set), names)
end

parent(fs::FeatureSet) = something(fs.__parent, fs)

###-----------------------------------------------------------------------------
### File API
###-----------------------------------------------------------------------------

function save(feature_set::FeatureSet; directory = ".")::Nothing
    path::String = joinpath(directory, filename(feature_set))
    @info "Created file" path
    save(path, feature_set)
    return nothing
end

function save(filename::AbstractString,
              feature_set::FeatureSet{L, N, F}
             )::Nothing where {L, N, F}
    h5open(filename, "w") do file
        file["id"] = id(feature_set) |> to_hdf5
        file["created_at"] = created_at(feature_set) |> to_hdf5
        file["labels"] = labels(feature_set) |> to_hdf5
        file["names"] = names(feature_set) |> to_hdf5
        fts = create_dataset(file,
                             "features",
                             datatype(F),
                             dataspace(size(feature_set)))
        fts[:, :] = features(feature_set)
    end

    return nothing
end

function load(::Type{FeatureSet},
              path::AbstractString;
              mmap::Bool = false
             )::FeatureSet
    return h5open(path, "r") do file
        @assert isvalid(FeatureSet, file)

        id = read(file, "id") |> UUID
        created_at = read(file, "created_at") |> DateTime
        features = mmap ? readmmap(file["features"]) : read(file, "features")
        labels = read(file, "labels")
        names = read(file, "names")
        return FeatureSet(labels, names, features; id, created_at)
    end
end

##------------------------------------------------------------------------------
## Miscellaneous functions
##------------------------------------------------------------------------------

function filename(feature_set::FeatureSet)::String
    return "$(id(feature_set)).hdf5"
end

function isvalid(::Type{FeatureSet}, path::AbstractString)::Bool
    return h5open(path, "r") do file
        return isvalid(FeatureSet, file)
    end
end

function isvalid(::Type{FeatureSet}, file::HDF5File)::Bool
    return ["id", "created_at", "labels", "names", "features"] ⊆ keys(file)
end

###-----------------------------------------------------------------------------
### Others
###-----------------------------------------------------------------------------

function merge(a::FS, b::FS)::FS where {FS <: FeatureSet}
    return a === b ? a :
        parent(a) === parent(b) ? merge_subarrays(a, b) :
        merge_(a, b)
end

function merge_subarrays(a::FeatureSet, b::FeatureSet)::FeatureSet
    (a_rows, a_cols) = parentindices(features(a))
    (b_rows, b_cols) = parentindices(features(b))
    @assert a_rows == b_rows
    unique_cols = unique!([a_cols; b_cols])
    unique_features = view(features(parent(a)), a_rows, unique_cols)
    unique_names = view(names(parent(a)), unique_cols)
    return FeatureSet(labels(a),
                      unique_names,
                      unique_features;
                      __parent = parent(a))
end

function merge_(a::FeatureSet, b::FeatureSet)::FeatureSet
    @assert labels(a) == labels(b)
    common_names::Vector = names(a) ∩ names(b)
    @assert(features(a)[:, resolve_name_index(a, common_names)] ==
        features(b)[:, resolve_name_index(b, common_names)],
            "Identically named features with different values!")

    only_b_names::Vector = setdiff(names(b), names(a))
    only_b_cols = resolve_name_index(b, only_b_names)
    return FeatureSet(labels(a),
                      vcat(names(a), only_b_names),
                      hcat(features(a), features(b)[:, only_b_cols]))
end

# TODO https://github.com/cursorinsight/FeatureScreening.jl/issues/12
"""
This function generates only per-label-BALANCED feature set.
"""
function rand(::Type{FeatureSet{L, N, F}},
              sample_count::Integer = 10,
              feature_count::Integer = 10;
              label_count::Integer = sample_count ÷ 5,
              center::Function = i -> ((i-1) / label_count + 1),
              place::Function = j -> 7j / feature_count,
              random::Function = (i, j) -> randn()
             )::FeatureSet{L, N, F} where {L, N <: Integer, F <: AbstractFloat}
    (d, r) = divrem(sample_count, label_count)
    @assert iszero(r)

    labels::Vector{L} = L.(repeat(1:label_count, inner = d))
    names::Vector{N} = N.(collect(1:feature_count))

    features::Matrix{F} =
        [center(i) * place(j) + random(i, j)
         for i in 1:sample_count, j in 1:feature_count]

    return FeatureSet(labels, names, features)
end

function rand(::Type{FeatureSet},
              sample_count::Integer = 10,
              feature_count::Integer = 10;
              kwargs...
             )::FeatureSet
    return rand(FeatureSet{Int, Int, Float64},
                sample_count,
                feature_count;
                kwargs...)
end

function __init__()
    @require_extensions
end

end # module FeatureSets
