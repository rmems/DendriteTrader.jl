# SPDX-License-Identifier: MIT OR Apache-2.0

@testset "Delta spike encoder" begin
    frame = FeatureFrame([
        FeatureRow(10, 1, 0, 0, 0, 0, 0),
        FeatureRow(20, 2, 2, -3, 0.2, 0, 10),
        FeatureRow(30, 3, 20, -20, 0.2, 0, 10),
    ])
    encoder = DeltaEncoder(scale = 1, threshold = 1, saturation = 4)
    encoded = encode!(encoder, frame)
    @test encoded.spikes[1, :] == zeros(Int8, 5)
    @test encoded.spikes[2, :] == Int8[2, -3, 0, 0, 4]
    @test encoded.spikes[3, :] == Int8[4, -4, 0, 0, 0]
    @test spike_density(encoded) ≈ 5 / 15
    reset_state!(encoder)
    @test encode!(encoder, frame).spikes == encoded.spikes
    @test_throws ArgumentError DeltaEncoder(scale = 0)
    @test_throws ArgumentError DeltaEncoder(threshold = -1)
    @test_throws ArgumentError DeltaEncoder(saturation = 128)
end
