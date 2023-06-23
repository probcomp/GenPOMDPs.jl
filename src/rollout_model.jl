function RolloutModel(pomdp::GenPOMDP, π::GenController)
    model = @eval begin
        @gen (static, diffs) function _init(params)
            state ~ ($p.init)(params)
            obs ~ ($p.obs)(state, params)

            control ~ controller($π.init_state, obs)
            
            return maketuple(state, obs, control)
        end
        @gen (static, diffs) function _step(t, prev, params)
            (prev_state, prev_obs, prev_control) = prev
            (action, πstate) = prev_control

            state ~ ($p.step)(prev_state, action, params)
            obs ~ ($p.obs)(state, params)

            control ~ controller(πstate, obs)

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
    @suppress Gen.load_generated_functions()

    return model
end