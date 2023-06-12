#=
A few TODOs:
1. Document the versions of `pf`, `pf_initializer`, `pf_updater` which accept functions
    that construct the particle filter params.
2. Think about if I want to remove or modify the old interface to `pf`, `pf_initializer`, and `pf_updater`,
    since as it is setup, the multiple dispatch behavior could be confusing to users.
=#

#=
Utilities for particle filtering in the generative functions
over trajectories exposed in GenPOMDPs.

The low-level particle filtering functionality is provided by
[GenParticleFilters.jl](https://github.com/probcomp/GenParticleFilters.jl).
=#
using GenParticleFilters: pf_initialize, pf_update!, get_ess, pf_resample!

"""
A Particle Filter is specified as a tuple of two functions:
1. initialize_particle_filter: POMDP, PARAMS -> PARTICLE_FILTER_STATE
2. update_particle_filter: PARTICLE_FILTER_STATE, OBSERVATION, ACTION -> PARTICLE_FILTER_STATE
"""
const ParticleFilter = Tuple{Function, Function}

### PF Observable which automatically updates on new (obs, action) ###

"""
Given an Observable of a pair (observation_sequence, action_sequence),
and a particle filter, returns an Observable of the particle filter state.

The particle filter state is updated whenever the (observation_sequence, action_sequence)
observable is updated. 

RESTRICTIONS:
- The observation sequence should always be one longer than the action sequence.
- The only update to the observations_and_actions observable should be to extend
  the observation sequence by 1, and the action sequence by 1.
  [If the observable is updated in other ways, errors may occur.]
"""
function pf_observable(
    pf::ParticleFilter,

    # An observable of a pair (observation_sequence, action_sequence).
    # Every time this observable is updated, it should be updated by
    # adding a new observation and a new action to these vectors.
    # [If the observable is updated in other ways, errors may occur.]
    # The particle filter will update on each such update.
    observations_and_actions
)
    (initialize_pf::Function, update_pf::Function) = pf

    # For checking correct usage:
    prev_obsact = observations_and_actions[]

    # Initialize
    obs0 = only(observations_and_actions[][1])
    pf_states = Observables.Observable([ initialize_pf(obs0) ])

    # Whenever a new observation and action become available:
    Observables.on(observations_and_actions) do (obss, actions)
        # Check correct usage:
        @assert length(obss) == length(actions) + 1 "Should have 1 more observation than action [due to initial observation]"
        @assert obss[1:end-1] == prev_obsact[1] "Observations should be extended by 1 exactly"
        @assert actions[1:end-1] == prev_obsact[2] "Actions should be extended by 1 exactly"
        prev_obsact = (obss, actions)

        # Update the particle filter state
        pf_state = pf_states[][end]
        updated = update_pf(pf_state, actions[end], obss[end])
        pf_states[] = push!(pf_states[], updated)
    end

    return pf_states
end

### Particle Filter via GenParticleFilters ###

"""
    pf(
        pomdp::POMDP, pomdp_params,
        pf_initialize_params::Tuple,
        pf_update_params::Tuple;
        pre_update = stratified_resample_if_ess_below_one_plus_onetenth_particlecount,
        post_update = (_ -> ())
    )

A particle filter for the given POMDP with the given pomdp_params, based on
the GenParticleFilters.jl library.

- pf_initialize_params controls the initialization of the particle filter:
  The PF is initialized via the call
  `GenParticleFilters.pf_initialize(model, args, observations, pf_initialize_params...)`.
- pf_update_params controls the particle filter update: The PF is updated via the call
  `GenParticleFilters.pf_update!(pf_state, new_args, new_observation, pf_update_params...)`.
- pre_update and post_update are optional kwargs.  They are functions
  that are called before and after each `pf_update!` call.
  They are called with the particle filter state as the only argument.
  They can be used, e.g., to include resampling and rejuvenation.

Examples of how to set the PF parameters:
- BOOTSTRAP Particle Filter:
    `pf_initialize_params = (n_particles,)`,
    `pf_update_params = ()`
- PF with custom proposal:
    `pf_initialize_params = (n_particles, initial_proposal, initial_proposal_args)`,
    `pf_update_params = (step_proposal, step_proposal_args)`
"""
function pf(pomdp, pomdp_params, pf_initialize_params, pf_update_params;
    pre_update = stratified_resample_if_ess_below_one_plus_onetenth_particlecount,
    post_update = (_ -> ())
)
    return (
        pf_initializer(pomdp, pomdp_params, pf_initialize_params...),
        pf_updater(pf_update_params...; pre_update, post_update)
    )
end
function pf(pomdp, pomdp_params, obs_to_pf_initialize_params::Function, act_obs_to_pf_update_params::Function;
    pre_update = stratified_resample_if_ess_below_one_plus_onetenth_particlecount,
    post_update = (_ -> ())
)
    return (
        pf_initializer(pomdp, pomdp_params, obs_to_pf_initialize_params),
        pf_updater(act_obs_to_pf_update_params; pre_update, post_update)
    )
end

function bootstrap_pf(pomdp, params, n_particles)
    return pf(pomdp, params, (n_particles,), ())
end

function pf_initializer(pomdp::GenPOMDP, pomdp_params, obs_to_pf_params::Function)
    controlled_trajectory_model = ControlledTrajectoryModel(pomdp)
    function initialize(obs0)
        pf_state = pf_initialize(
            controlled_trajectory_model,
            (0, [], pomdp_params),
            nest_choicemap(obs0, obs_addr(0)),
            obs_to_pf_params(obs0)...
        )
        return pf_state    end

    return initialize
end
pf_initializer(pomdp::GenPOMDP, pomdp_params, pf_initialize_params...) =
    pf_initializer(pomdp, pomdp_params, (_ -> pf_initialize_params))

# function pf_initializer(pomdp::GenPOMDP, pomdp_params, pf_initialize_params...)
    # function initialize(obs0)
    #     # gf = ControlledTrajectoryModel(pomdp)
    #     # display(nest_choicemap(obs0, obs_addr(0)))
    #     pf_state = pf_initialize(
    #         controlled_trajectory_model,
    #         (0, [], pomdp_params),
    #         nest_choicemap(obs0, obs_addr(0)),
    #         pf_initialize_params...
    #     )
    #     return pf_state
    # end
# end

function pf_updater(act_obs_to_pf_update_params::Function;
    pre_update = stratified_resample_if_ess_below_one_plus_onetenth_particlecount,
    post_update = (_ -> ())
)
    function update(pf_state, newaction, newobs)
        (T, oldactions, pomdp_params) = get_args(pf_state.traces[1])
        new_pf_state = copy(pf_state)

        pre_update(new_pf_state) # E.g. resample
        
        actions = vcat(oldactions, [newaction])
        pf_update!(
            new_pf_state,
            (T + 1, actions, pomdp_params),
            (Gen.IntDiff(1), # T has changed
                
                # HACK: say that the actions have not changed.
                # (In this specific case, the correct update
                # will occur if we tell Gen this.
                # This will help performance.
                # Eventually we need to add a better way for Gen
                # to achieve this performance gain.)
                NoChange(),

                # Parameters have not changed
                NoChange()
            ),
            nest_choicemap(newobs, obs_addr(T + 1)),

            act_obs_to_pf_update_params(newaction, newobs)...
        )
        post_update(new_pf_state) # E.g. rejuvenate

        return new_pf_state
    end
    
    return update
end
pf_updater(pf_update_params...; kwargs...) = pf_updater(((_, _) -> pf_update_params)...; kwargs...)
# function pf_updater(pf_update_params...;
#     pre_update = stratified_resample_if_ess_below_onefifth_particlecount,
#     post_update = (_ -> ())
# )
#     function update(pf_state, newaction, newobs)
#         (T, oldactions, pomdp_params) = get_args(pf_state.traces[1])
#         new_pf_state = copy(pf_state)

#         pre_update(new_pf_state) # E.g. resample
        
#         actions = vcat(oldactions, [newaction])
#         pf_update!(
#             new_pf_state,
#             (T + 1, actions, pomdp_params),
#             (Gen.IntDiff(1), # T has changed
                
#                 # HACK: say that the actions have not changed.
#                 # (In this specific case, the correct update
#                 # will occur if we tell Gen this.
#                 # This will help performance.
#                 # Eventually we need to add a better way for Gen
#                 # to achieve this performance gain.)
#                 NoChange(),

#                 # Parameters have not changed
#                 NoChange()
#             ),
#             nest_choicemap(newobs, obs_addr(T + 1)),

#             pf_update_params...
#         )
#         post_update(new_pf_state) # E.g. rejuvenate

#         return new_pf_state
#     end

#     return update
# end
function stratified_resample_if_ess_below_onefifth_particlecount(pf_state)
    if get_ess(pf_state) < 0.2 * length(get_traces(pf_state))
        pf_resample!(pf_state, :stratified)
    end
end
function stratified_resample_if_ess_below_one_plus_onetenth_particlecount(pf_state)
    if get_ess(pf_state) < 1 + 0.1 * length(get_traces(pf_state))
        pf_resample!(pf_state, :stratified)
    end
end