# SPDX-License-Identifier: MIT OR Apache-2.0

module Spikes

using ..DendriteTrader: FeatureFrame, FeatureRow
import ..DendriteTrader: reset_state!

export AbstractSpikeEncoder, SpikeFrame, DeltaEncoder, encode!, reset_state!, spike_density

include("Interface.jl")
include("DeltaEncoder.jl")

end # module
