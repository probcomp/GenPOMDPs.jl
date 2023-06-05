function ControlledTrajectoryModel(p::GenPOMDP)
    return (
        @gen function ControlledTrajectoryModel(action_sequence, params)
            states, obss, utilities = [], [], []
            
            state = {:state0} ~ p.init(params)
            push!(states, state)

            for (t, action) in enumerate(action_sequence)
                state = {:steps => t => :state} ~ p.step(state, action, params)
                obs = {:steps => t => :obs} ~ p.obs(state, params)
                utility = p.utility(state, action)
                push!(states, state)
                push!(obss, obs)
                push!(utilities, utility)
            end

            return (states, obss, utilities)
        end
    )
end

# I want to be able to write:
function ControlledTrajectoryModel(p::GenPOMDP)
    return (
        @gen (static, diffs) function ControlledTrajectoryModel(action_sequence, params)
            states, obss, utilities = [], [], []
            
            state = {:steps => 0} ~ p.init(params)
            push!(states, state)

            for (t, action) in enumerate(action_sequence)
                state = {:steps => t => :state} ~ p.step(state, action, params)
                obs = {:steps => t => :obs} ~ p.obs(state, params)
                utility = p.utility(state, action, params)
                push!(states, state)
                push!(obss, obs)
                push!(utilities, utility)
            end

            return (states, obss, utilities)
        end
    )
end

# Next best thing:
function ControlledTrajectoryModel(p::GenPOMDP)
    return (
        @gen (static, diffs) function ControlledTrajectoryModel(action_sequence, params)
            state0 = {:state0} ~ p.init(params)
            obs0 = {:obs0} ~ p.obs(state0, params)

            states ~ Unfold(
                # Call the step model, dropping
                # the time argument Unfold will try to pass in
                argmap_gf(p.step, (t, args...) -> args)
            )(length(action_sequence), state0, action_sequence, params)

            obss ~ Map(p.obs)(states, fill(params, length(states)))

            utilities = map(p.utility, zip(states, action_sequence, fill(params, length(states))))

            return (vcat([state0], states), vcat([obs0], obss), utilities)
        end
    )
end
# Or maybe it's better to write it as:










# ### TODO: below here needs work! ###
# """
# An observable of a pair (observation_sequence, action_sequence),
# describing the sequence of actions and received observations in a
# environment trace.
# (observation_sequence[t] is the observation received before performing action_sequence[t])
# """
# function obs_action_observable(ground_truth_tr::Observables.Observable{<:Gen.Trace})
#     lift(ground_truth_tr) do tr
#         states, obss, utilities = get_retval(tr)
#         actions, params = get_args(tr)
    
#         return (obss, actions)
#     end
# end

# # -> PF observable, which updates whenever the (obs, action) observable updates.
# #    The function to construct this should also see params for the "world model", and
# #    should see a POMDP.
# function pf_observable(obs_action_observable, initialize_pf::Function, update_pf::Function)
#     # A vector, of the particle filter state at each timestep.
#     initial_obs, initial_action = obs_action_observable[][1][1], obs_action_observable[][2][1]
#     pf_states = Observable([ initialize_pf() ])
    
#     # Whenever the trace is updated, update the particle filter state,
#     # and append it to the state list
#     Observables.on(obs_action_observable) do (obss, actions)
#         pf_state = pf_states[][end]
#         updated = update_pf(pf_state, obss[end], actions[end])
#         pf_states[] = push!(pf_states[], updated)
#     end

#     return pf_states
# end
# function pf_observable(obs_action_observable, params, mental_controlled_trajectory_model::Gen.GenerativeFunction, n_particles)
#     function initialize_pf(obs, action)
#         pf_initialize(mental_controlled_trajectory_model, params, obs_choicemap, n_particles)
#     end

#     # TODO
# end