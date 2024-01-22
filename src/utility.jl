function undiscounted_utility(pomdp::GenPOMDP, trace::Gen.Trace)
    return sum(pomdp.utility(s, a) for (s, a) in zip(state_sequence(trace), action_sequence(trace)))
end