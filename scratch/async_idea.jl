# to have within the function `pf_observable`:

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
