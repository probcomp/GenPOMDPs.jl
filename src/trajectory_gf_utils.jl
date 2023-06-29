state_addr(t) = t == 0 ? (:init => :state) : (:steps => t => :state)
obs_addr(t) = t == 0 ? (:init => :obs) : (:steps => t => :obs)

state_addr(t, subaddr) = nest_at(state_addr(t), subaddr)
obs_addr(t, subaddr) = nest_at(obs_addr(t), subaddr)

# The observations are the _choicemaps_ of the obs generative functions
observation_sequence(tr) = [get_submap(get_choices(tr), obs_addr(t)) for t=0:get_args(tr)[1]]

# The states are the _return value_ of the step generative functions
state_sequence(tr) = [tr[state_addr(t)] for t=0:get_args(tr)[1]]

observation_retval_sequence(tr) = [tr[obs_addr(t)] for t=0:get_args(tr)[1]]

function action_sequence(tr)
    if length(get_args(tr)) == 2 # `tr` from ControlledTrajectoryModel
        return get_args(tr)[2]
    else # `tr` from RolloutModel
        return map(x -> x[1], control_sequence(tr))
    end
end

# For RolloutModels only:
control_addr(t) = t == 0 ? (:init => :control) : (:steps => t => :control)
control_choicemap_sequence(tr) = [get_submap(get_choices(tr), control_addr(t)) for t=0:get_args(tr)[1]]

control_sequence(tr) = [tr[control_addr(t)] for t=0:get_args(tr)[1]]
controllerstate_sequence(tr) = map(x -> x[2], control_sequence(tr))