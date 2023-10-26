###-----------------------------------------------------------------------------
### Copyright (C) The FeatureSets.jl contributors
###
### SPDX-License-Identifier: MIT
###-----------------------------------------------------------------------------

module TablesExt

###=============================================================================
### Imports
###=============================================================================

using FeatureSets: AbstractFeatureSet, features, names
using Tables: Schema, table

import Tables: columnaccess, columns, istable, rowaccess, rows
import Tables: schema, subset

###=============================================================================
### Implementation
###=============================================================================

istable(::Type{<: AbstractFeatureSet}) = true
rowaccess(::Type{<: AbstractFeatureSet}) = true
columnaccess(::Type{<: AbstractFeatureSet}) = true

rows(fs::AbstractFeatureSet) = fs |> astable |> rows
columns(fs::AbstractFeatureSet) = fs |> astable |> columns

function schema(fs::AbstractFeatureSet{T, N}) where {T, N}
    return Schema(Symbol.(names(fs)), fill(T, size(fs, 2)))
end

function subset(fs::AbstractFeatureSet, inds; viewhint = nothing)
    return viewhint === false ? getindex(fs, inds, :) : view(fs, inds, :)
end

##------------------------------------------------------------------------------
## Internals
##------------------------------------------------------------------------------

function astable(fs::AbstractFeatureSet)
    return table(features(fs); header = Symbol.(names(fs)))
end

end # module TablesExt
