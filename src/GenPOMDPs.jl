module GenPOMDPs

using Gen

# Julia library for "Observables": mutable values that can be subscribed to
# and which notify their subscribers when they change.
# We use this to enable real-time and interactive visualizations.
import Observables

export GenPOMDP, ControlledTrajectoryModel

struct GenPOMDP
    # Generative function accepting params and returning an initial state
    init    :: Gen.GenerativeFunction

    # Generative function accepting (state, action, params) as input, and returning
    # a new state
    step    :: Gen.GenerativeFunction

    # Generative function accepting (state, params) as input, whose choicemap
    # is the observation.
    obs     :: Gen.GenerativeFunction

    # Function accepting a state and action as input, and returning
    # the utility (ie. reward) of that (state, action) pair
    utility :: Function
end

# Various utilities for working with Gen,
# some of which should concievably be ported to Gen.jl eventually
include("gen_utils.jl")

# Utilities for working with traces of the generative functions over trajectories
# which are exposed in this module.
include("trajectory_gf_utils.jl")

# Generative functions over POMDP trajectories, given a fixed action sequence
include("controlled_trajectory.jl")

# Utilities for constructing particle filters for the generative functions
# over trajectories exposed in this module.    
include("particle_filter.jl")

end # module GenPOMDPs
