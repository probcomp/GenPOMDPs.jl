function RolloutModel(p::GenPOMDP, π::Controller)
    model = @eval begin
        @gen (static, diffs) function _init(params)
            state ~ ($p.init)(params)
            obs ~ ($p.obs)(state, params)

            control ~ $π.controller($π.init_state, obs, params)
            
            return maketuple(state, obs, control)
        end

        @gen (static, diffs) function _step(t, prev, params)
            (prev_state, prev_obs, prev_control) = prev

            (action, πstate) = prev_control

            state ~ ($p.step)(prev_state, action, params)
            obs ~ ($p.obs)(state, params)

            control ~ $π.controller(πstate, obs, params)

            return maketuple(state, obs, control)
        end

        @gen (static, diffs) function _RolloutModel(T, params)
            init ~ _init(params)
            steps ~ Unfold(_step)(T, init, params)

            # Return:
            # init - (state0, obs0, control0)
            # steps - vector of (state_t, obs_t, control_t), for each t ≥ 1
            return maketuple(init, steps)
        end
    end

    # I believe we need to do this for compilation to occur properly.
    # Use @suppress since there is currently a deprecation warning on Gen.load_generated_functions.
    # Before this deprecation goes through we will need to find some other way to get compilation
    # to work in this sort of context.
    # @suppress Gen.load_generated_functions()

    return model
end

#=
TODO: Elsewhere, I have the convention that, officially, the _choicemap_ of the observation model
is the true observation.  But here, I'm passing the return value of the observation into the controller.
Maybe this is fine -- if so I should document it.
But is there a better design?
=#

#=
TODO: add support for more automatic episode termination.
One implementation strategy: if the state is ever `nothing`, then
have all remaining states, observations, and controls be `nothing`.

Could do this using the `Switch` combinator, in conjunction with:
@gen (static, diffs) function _terminal_step(t, prev, params)
    # If the episode has terminated...
    if isnothing(prev_state)
        state   ~ exactly(nothing)
        obs     ~ exactly(nothing)
        control ~ exactly((nothing, nothing))
        return maketuple(state, obs, control)
    end
end

This may require improving the Switch combinator.
=#