###-----------------------------------------------------------------------------
### Copyright (C) The FeatureSets.jl contributors
###
### SPDX-License-Identifier: MIT
###-----------------------------------------------------------------------------

module FeatureSetsTest

###=============================================================================
### Imports
###=============================================================================

using Test

# Dependencies
using Base: ReshapedArray
using DecisionTree: Ensemble as RandomForest
using DecisionTree: apply_forest, build_forest, confusion_matrix
using HDF5: File, h5open, ishdf5
using Tables

# Tested API
using FeatureSets: FeatureSet
using FeatureSets: features, labels, names
using FeatureSets: id, isvalid, load, save

# Utilities
using ..Fixtures: fixture

###=============================================================================
### Tests
###=============================================================================

@testset "`FeatureSet` function" begin
    let feature_set = FeatureSet([1, 1, 1, 2, 2, 2, 3, 3, 3],
                                 "feature" .* ('1':'4'),
                                 [10 11 12 13
                                  11 11 12 13
                                  13 12 11 12
                                  21 20 23 24
                                  22 21 22 23
                                  21 22 22 22
                                  33 32 31 32
                                  31 34 31 34
                                  39 30 30 32])

        # scalar indexing returns feature value
        @test feature_set[2, "feature3"] == 12
        @test feature_set[8, "feature4"] == 34

        # combination of scalar and vector/colon indexing returns vector
        @test feature_set[3, ["feature2", "feature3"]] == [12, 11]
        @test feature_set[4, :] == [21, 20, 23, 24]
        @test feature_set[5:6, "feature1"] == [22, 21]
        @test feature_set[:, "feature1"] == [10, 11, 13, 21, 22, 21, 33, 31, 39]

        # indexing with two vectors returns FeatureSet (subset)
        @testset for (isview, feature_subset) in
            [(false, feature_set[:, ["feature2", "feature3", "feature4"]])
             (true, @view feature_set[:, ["feature2", "feature3", "feature4"]])]

            @test feature_subset isa FeatureSet
            @test labels(feature_subset) == [1, 1, 1, 2, 2, 2, 3, 3, 3]
            @test names(feature_subset) == ["feature2", "feature3", "feature4"]
            @test features(feature_subset) == [11 12 13
                                               11 12 13
                                               12 11 12
                                               20 23 24
                                               21 22 23
                                               22 22 22
                                               32 31 32
                                               34 31 34
                                               30 30 32]

            @test isview == (feature_subset.__parent isa FeatureSet)
            @test isview == (labels(feature_subset) isa SubArray)
            @test isview == (names(feature_subset) isa SubArray)
            @test isview == (features(feature_subset) isa SubArray)

            @test [(1, [11, 12, 13]),
                   (1, [11, 12, 13]),
                   (1, [12, 11, 12]),
                   (2, [20, 23, 24]),
                   (2, [21, 22, 23]),
                   (2, [22, 22, 22]),
                   (3, [32, 31, 32]),
                   (3, [34, 31, 34]),
                   (3, [30, 30, 32])
                  ] == collect(eachrow(feature_subset))

            @testset for feature_subset2 in
                [feature_subset[1:5, ["feature2", "feature3"]]
                 @view feature_subset[1:5, ["feature2", "feature3"]]]
                @test feature_subset2 isa FeatureSet
                @test labels(feature_subset2) == [1, 1, 1, 2, 2]
                @test names(feature_subset2) == ["feature2", "feature3"]
                @test features(feature_subset2) == [11 12;
                                                    11 12;
                                                    12 11;
                                                    20 23;
                                                    21 22]
                @test [(1, [11, 12]),
                       (1, [11, 12]),
                       (1, [12, 11]),
                       (2, [20, 23]),
                       (2, [21, 22])
                      ] == collect(eachrow(feature_subset2))
            end

            feature_subset3 = feature_subset[:, :]
            @test feature_subset3 isa FeatureSet
            @test feature_subset == feature_subset3

            feature_subset4 = @view feature_subset3[1:end, :]
            @test feature_subset4 isa FeatureSet
            @test feature_subset3 == feature_subset4
        end
    end

    let feature_set = rand(FeatureSet, 80, 30)
        @test feature_set isa FeatureSet{Int, Int, Float64}
        @test size(feature_set) == (80, 30)
        @test axes(feature_set, 1) == 1:80
        @test axes(feature_set, 2) == 1:30
        @test axes(feature_set) == (1:80, 1:30)
    end

    let feature_set = rand(FeatureSet{Char, Int, Float64}, 80, 30)
        @test feature_set isa FeatureSet{Char, Int, Float64}
        @test size(feature_set) == (80, 30)
        @test axes(feature_set, 1) == 1:80
        @test axes(feature_set, 2) == 1:30
        @test axes(feature_set) == (1:80, 1:30)
    end
end

@testset "merge" begin
    fs1 = FeatureSet(1:2, [:a, :b, :c], [1 2 3; 4 5 6])

    # merging subsets (views and copies)
    let merged = merge(view(fs1, 1:2, [:a]), view(fs1, 1:2, [:b, :c]))
        @test merged.__parent === fs1
        @test merged == fs1
    end
    let merged = merge(fs1[1:2, [:a]], fs1[1:2, [:b, :c]])
        @test merged.__parent === nothing
        @test merged == fs1
    end

    # merging overlapping subsets (views and copies)
    let merged = merge(view(fs1, 1:2, [:a, :b]), view(fs1, 1:2, [:b, :c]))
        @test merged.__parent === fs1
        @test merged == fs1
    end
    let merged = merge(fs1[1:2, [:a, :b]], fs1[1:2, [:b, :c]])
        @test merged.__parent === nothing
        @test merged == fs1
    end

    # merging with self and merging two identical feature sets
    @test merge(fs1, fs1) == fs1
    @test merge(fs1, deepcopy(fs1)) == fs1

    # merging two disjunct feature sets
    let fs2 = FeatureSet(1:2, [:d, :e], [7 8; 9 10]),
        merged = merge(fs1, fs2)
        @test labels(merged) == 1:2
        @test names(merged) == [:a, :b, :c, :d, :e]
        @test features(merged) == [1 2 3 7 8; 4 5 6 9 10]
    end

    # merging two feature sets with shared features
    let fs2 = FeatureSet(1:2, [:a, :b, :d], [1 2 7; 4 5 8]),
        merged = merge(fs1, fs2)
        @test labels(merged) == 1:2
        @test names(merged) == [:a, :b, :c, :d]
        @test features(merged) == [1 2 3 7; 4 5 6 8]
    end

    # merging two disjunct subsets
    let fs2 = FeatureSet(1:2, [:d, :e], [7 8; 9 10]),
        merged = merge(fs1[1:2, [:a, :b]], fs2[1:2, [:d]])
        @test labels(merged) == 1:2
        @test names(merged) == [:a, :b, :d,]
        @test features(merged) == [1 2 7; 4 5 9]
    end
end

const FEATURE_SETS =
    ["feature set" => rand(FeatureSet, 50, 30; label_count = 10),
     "feature subset" => rand(FeatureSet, 50, 30; label_count = 10)[:, 1:5]]

@testset "Save and load $name" for (name, feature_set) in FEATURE_SETS
    mktempdir() do directory::String
        # Save
        @test_logs (:info, "Created file") save(feature_set; directory)

        path = "$directory/$(id(feature_set)).hdf5"

        # File storage content
        @test isfile(path)
        @test ishdf5(path)
        @test isvalid(FeatureSet, path)

        # Load
        let loaded::FeatureSet = load(FeatureSet, path)
            @test loaded isa FeatureSet
            @test loaded == feature_set
            @test size(loaded) == size(feature_set)
            @test features(loaded) isa Matrix{Float64}

            for (label, features) in loaded
                @test label isa Int
                @test label in labels(feature_set)
                @test features isa AbstractVector{Float64}
                @test length(features) == size(feature_set, 2)
            end
        end

        # Save view
        fsv = view(feature_set, 1:2, 1:2)
        @assert parent(fsv) == feature_set
        @test_logs (:info, "Created file") save(fsv; directory)

        path = "$directory/$(id(fsv)).hdf5"

        @test isfile(path)
        @test ishdf5(path)
        @test isvalid(FeatureSet, path)

        # Load
        let loaded = load(FeatureSet, path)
            @test loaded isa FeatureSet
            @test loaded.__parent == nothing
            @test loaded == fsv
        end
    end
end

@testset "Mmapped IO" begin
    mktemp() do path, io
        close(io)

        # anything smaller than 300×300 does not get memory mapped
        feature_set = rand(FeatureSet, 300, 300)
        save(path, feature_set)

        # File storage content
        @test isfile(path)
        @test ishdf5(path)
        @test isvalid(FeatureSet, path)

        loaded::FeatureSet = load(FeatureSet, path; mmap = true)
        @test feature_set isa FeatureSet
        @test size(loaded) == size(feature_set)
        @test features(loaded) isa ReshapedArray{Float64, 2}
    end
end

@testset "DecisionTree API" begin
    fs = fixture(:feature_set)
    forest = build_forest(fs)
    @test forest isa RandomForest

    preds = apply_forest(forest, fs)
    @test preds isa Vector{Symbol}

    cm = confusion_matrix(forest, fs)
    @test cm.matrix isa Matrix{Int64}
    @test cm.accuracy >= 0.8
    @test cm.kappa >= 0.8
end

@testset "Tables API" begin
    fs_names = "feature_" .* ('1':'9')
    fs = FeatureSet(rand(1:10, 10), fs_names, rand(10, 9))
    # test that the MatrixTable `istable`
    @test Tables.istable(typeof(fs))
    @test Tables.schema(fs) isa Tables.Schema
    # test that it defines row access
    @test Tables.rowaccess(typeof(fs))
    @test applicable(iterate, Tables.rows(fs))
    # test that it defines column access
    @test Tables.columnaccess(typeof(fs))
    @test applicable(iterate, Tables.columns(fs))
    let cols = Tables.columns(fs), value = fs[:, "feature_1"]
        @test cols.feature_1 == value
        # test our `Tables.AbstractColumns` interface methods
        @test Tables.getcolumn(cols, :feature_1) == value
        @test Tables.getcolumn(cols, 1) ==  value
        @test Tables.columnnames(cols) == Symbol.(fs_names)
    end
    # test that we can access the first row by index
    let row = first(Tables.rows(fs)), value = fs[1, "feature_1"]
        @test eltype(fs) == typeof(first(row))
        @test row.feature_1 == value
        # test our `Tables.AbstractRow` interface methods
        @test Tables.getcolumn(row, :feature_1) == value
        @test Tables.getcolumn(row, 1) == value
        @test propertynames(row) == Symbol.(fs_names)
        @test Tables.columnnames(row) == Symbol.(fs_names)
    end
    @test Tables.subset(fs, 3:5) == fs[3:5, :]
end

end # module FeatureSetsTest
