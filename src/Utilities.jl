###-----------------------------------------------------------------------------
### Copyright (C) The FeatureSets.jl contributors
###
### SPDX-License-Identifier: MIT
###-----------------------------------------------------------------------------

module Utilities

###=============================================================================
### Exports
###=============================================================================

export @unimplemented, to_hdf5

###=============================================================================
### Imports
###=============================================================================

using Dates: DateTime
using MacroTools: combinedef, rmlines, splitdef
using UUIDs: UUID

###=============================================================================
### Implementation
###=============================================================================

macro unimplemented(function_definition::Expr)
    def = splitdef(function_definition)
    @assert(rmlines(def[:body]) == Expr(:block), "Function definition of " *
        "@unimplemented $(def[:name]) has a non-empty body!")
    def[:body] = :(error("Unimplemented method: ",
                         $(string(function_definition.args[1]))))
    return esc(combinedef(def))
end

"""
CAUTION! This function will create a new array if the input was an array view.
"""
function to_hdf5(x::SubArray{_T, _N, A})::A where {_T, _N, A}
    return copy(x)
end

function to_hdf5(x::AbstractRange)::Vector
    return collect(x)
end

function to_hdf5(x::UUID)::String
    return string(x)
end

function to_hdf5(x::DateTime)::String
    return string(x)
end

function to_hdf5(x)
    return x
end

end # module Utilities
