###-----------------------------------------------------------------------------
### Copyright (C) The FeatureSets.jl contributors
###
### SPDX-License-Identifier: MIT
###-----------------------------------------------------------------------------

module Fixtures

###=============================================================================
### Exports
###=============================================================================

export fixture

###=============================================================================
### Imports
###=============================================================================

using FeatureSets: FeatureSet

###=============================================================================
### Implementation
###=============================================================================

fixture(arg::Symbol) = fixture(Val(arg))

function fixture(::Val{:feature_set})
    return FeatureSet(fixture(:y), fixture(:names), fixture(:X))
end

function fixture(::Val{:y})
    return [:a, :a, :a, :a, :a,
            :b, :b, :b, :b, :b,
            :c, :c, :c, :c, :c,
            :d, :d, :d, :d, :d,
            :e, :e, :e, :e, :e]
end

function fixture(::Val{:names})
    return ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11"]
end

function fixture(::Val{:X})
    X::Matrix{Float64} =
        [0.0 -0.2 -0.4 -0.6 -0.8 -1.0 -1.2 -1.4 -1.6 -1.8 -2.0;
         0.0 -0.2 -0.4 -0.6 -0.8 -1.0 -1.2 -1.4 -1.6 -1.8 -2.0;
         0.0 -0.2 -0.4 -0.6 -0.8 -1.0 -1.2 -1.4 -1.6 -1.8 -2.0;
         0.0 -0.2 -0.4 -0.6 -0.8 -1.0 -1.2 -1.4 -1.6 -1.8 -2.0;
         0.0 -0.2 -0.4 -0.6 -0.8 -1.0 -1.2 -1.4 -1.6 -1.8 -2.0;
         0.0 -0.1 -0.2 -0.3 -0.4 -0.5 -0.6 -0.7 -0.8 -0.9 -1.0;
         0.0 -0.1 -0.2 -0.3 -0.4 -0.5 -0.6 -0.7 -0.8 -0.9 -1.0;
         0.0 -0.1 -0.2 -0.3 -0.4 -0.5 -0.6 -0.7 -0.8 -0.9 -1.0;
         0.0 -0.1 -0.2 -0.3 -0.4 -0.5 -0.6 -0.7 -0.8 -0.9 -1.0;
         0.0 -0.1 -0.2 -0.3 -0.4 -0.5 -0.6 -0.7 -0.8 -0.9 -1.0;
         0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0;
         0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0;
         0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0;
         0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0;
         0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0;
         0.0 +0.1 +0.2 +0.3 +0.4 +0.5 +0.6 +0.7 +0.8 +0.9 +1.0;
         0.0 +0.1 +0.2 +0.3 +0.4 +0.5 +0.6 +0.7 +0.8 +0.9 +1.0;
         0.0 +0.1 +0.2 +0.3 +0.4 +0.5 +0.6 +0.7 +0.8 +0.9 +1.0;
         0.0 +0.1 +0.2 +0.3 +0.4 +0.5 +0.6 +0.7 +0.8 +0.9 +1.0;
         0.0 +0.1 +0.2 +0.3 +0.4 +0.5 +0.6 +0.7 +0.8 +0.9 +1.0;
         0.0 +0.2 +0.4 +0.6 +0.8 +1.0 +1.2 +1.4 +1.6 +1.8 +2.0;
         0.0 +0.2 +0.4 +0.6 +0.8 +1.0 +1.2 +1.4 +1.6 +1.8 +2.0;
         0.0 +0.2 +0.4 +0.6 +0.8 +1.0 +1.2 +1.4 +1.6 +1.8 +2.0;
         0.0 +0.2 +0.4 +0.6 +0.8 +1.0 +1.2 +1.4 +1.6 +1.8 +2.0;
         0.0 +0.2 +0.4 +0.6 +0.8 +1.0 +1.2 +1.4 +1.6 +1.8 +2.0]

    return X .+ randn(size(X))
end

end # module
