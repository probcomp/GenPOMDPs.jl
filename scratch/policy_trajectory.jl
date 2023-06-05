#=
This is all VERY WIP!

But the idea is that this file lets us get a generative function which is a distribution
over the trajectory rollouts resulting from a given policy under a given POMDP.
=#

function PolicyTrajectoryModel(p::GenPOMDP, π)
    @gen (static, diffs) function _init(env_params, initial_policy_state)
        state ~ p.init(env_params)
        obs ~ p.obs(state, env_params)

        return ((state, obs), initial_policy_state)
    end

    @gen (static, diffs) function _step(t, prevstep, params)
        (env_params, policy_params) = params
        ((prev_world_state, prev_obs), prev_policy_state) = prevstep

        (action, policystate) = {:choice} ~ π(prevobs, prev_policy_state, policy_params)
        
        state ~ p.step(prevstate, action, env_params)
        obs ~ p.obs(state, env_params)

        return ((state, obs), policystate)
    end

    @gen (static, diffs) function PolicyTrajectoryModel(T, pomdp_params, policy_params, initial_policy_state)
        init ~ _init(pomdp_params, initial_policy_state)
        step ~ Unfold(_step)(T, init, (pomdp_params, policy_params))
    end
end