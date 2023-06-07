"""
    pf(pomdp::POMDP, pomdp_params, pf_initialize_params::Tuple, pf_update_params::Tuple)


"""
function pf(pomdp, pomdp_params, pf_initialize_params, pf_update_params)
    return (
        pf_initializer(pomdp, pomdp_params, pf_initialize_params...),
        pf_updater(pf_update_params...)
    )
end
function bootstrap_pf(pomdp, params, n_particles)
    return pf(pomdp, params, (n_particles,), ())
end

function pf_initializer(pomdp::GenPOMDP, model_params, pf_initialize_params...)
    controlled_trajectory_model = ControlledTrajectoryModel(pomdp)

    function initialize(obs0)
        # gf = ControlledTrajectoryModel(pomdp)
        # display(nest_choicemap(obs0, obs_addr(0)))
        pf_state = pf_initialize(
            controlled_trajectory_model,
            (0, [], params),
            nest_choicemap(obs0, obs_addr(0)),
            pf_initialize_params...
        )
        return pf_state
    end

    return initialize
end
function pf_updater(pre_update::Function, post_update::Function, pf_update_params...)
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
        nest_choicemap(newobs, obs_addr(T + 1)),

        pf_update_params...
    )
    post_update(new_pf_state) # E.g. rejuvenate

    return new_pf_state
end
# A reasonable default particle updater:
# Stratified resampling if the ESS falls below a threshold,
# then a PF update (controlled by given parameters).
function pf_updater(pf_update_params...)
    function maybe_resample!(pf_state)
        n_particles = length(get_traces(pf_state))
        if get_ess(pf_state) < ess_threshold_fraction * n_particles
            pf_resample!(pf_state, :stratified)
        end
    end

    return bootstrap_pf_updater(
        maybe_resample!, # Maybe resample before the bootstrap update
        (_ -> ()), # Don't do anything after the bootstrap update
        pf_update_params...
    )
end