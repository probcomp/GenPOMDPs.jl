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
    
    _pending_updates = []
    _processing = false
    function process_updates!()
        _processing = true

        (obs, act) = deleteat!(_pending_updates, 1)

        pf_state = pf_states[][end]
        updated = update_pf(pf_state, actions[end], obss[end])
        pf_states[] = push!(pf_states[], updated)

        if length(_pending_updates) > 0
            process_updates!()
        else
            _processing = false
        end
    end

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
        # push!(_pending_updates, (obss, actions))
        # @async process_updates!()
    end

    return pf_states
end

### Bootstrap PF ###

function bootstrap_pf(pomdp, params, n_particles; updater_kwargs...)
    return (
        bootstrap_pf_initializer(pomdp, params, n_particles),
        bootstrap_pf_updater(; updater_kwargs...)
    )
end

function bootstrap_pf_initializer(pomdp::GenPOMDP, params, n_particles;
    controlled_trajectory_model = ControlledTrajectoryModel(pomdp)    
)
    function initialize(obs0)
        # gf = ControlledTrajectoryModel(pomdp)
        # display(nest_choicemap(obs0, obs_addr(0)))
        pf_state = pf_initialize(
            controlled_trajectory_model,
            (0, [], params),
            nest_choicemap(obs0, obs_addr(0)),
            n_particles
        )
        return pf_state
    end

    return initialize
end

function bootstrap_pf_updater(pre_update, post_update)
    function update(pf_state, newaction, newobs)
        (T, oldactions, params) = get_args(pf_state.traces[1])
        new_pf_state = copy(pf_state)

        pre_update(new_pf_state) # E.g. resample

        actions = vcat(oldactions, [newaction])
        pf_update!(
            new_pf_state,
            (T + 1, actions, params),
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
            nest_choicemap(newobs, obs_addr(T + 1))
        )
        post_update(new_pf_state) # E.g. rejuvenate

        return new_pf_state
    end

    return update
end
# A reasonable default: boostrap PF with stratified resampling
# when the ESS is low.
function bootstrap_pf_updater(; ess_threshold_fraction = 1/5)
    function maybe_resample!(pf_state)
        n_particles = length(get_traces(pf_state))
        if get_ess(pf_state) < ess_threshold_fraction * n_particles
            pf_resample!(pf_state, :stratified)
        end
    end

    return bootstrap_pf_updater(
        maybe_resample!, # Maybe resample before the bootstrap update
        (_ -> ()) # Don't do anything after the bootstrap update
    )
end