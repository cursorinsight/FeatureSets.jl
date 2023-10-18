###-----------------------------------------------------------------------------
### Copyright (C) The FeatureSets.jl contributors
###
### SPDX-License-Identifier: MIT
###-----------------------------------------------------------------------------

###=============================================================================
### Implementation
###=============================================================================

using Test

using Aqua: test_all as aqua
using Random: seed!

# pad test summaries to equal length
Test.get_alignment(::Test.DefaultTestSet, ::Int) = 30

# fixed random seed
seed!(1)

###=============================================================================
### Tests
###=============================================================================

@testset "Aqua" begin
    import FeatureSets
    aqua(FeatureSets)
end

include("Fixtures.jl")
include("FeatureSetsTest.jl")
