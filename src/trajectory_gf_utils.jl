state_addr(t) = t == 0 ? (:init => :state) : (:steps => t => :state)
obs_addr(t) = t == 0 ? (:init => :obs) : (:steps => t => :obs)

state_addr(t, subaddr) = nest_at(state_addr(t), subaddr)
obs_addr(t, subaddr) = nest_at(obs_addr(t), subaddr)

# The observations are the _choicemaps_ of the obs generative functions
observation_sequence(tr) = [get_submap(get_choices(tr), obs_addr(t)) for t=0:get_args(tr)[1]]

# The states are the _return value_ of the step generative functions
state_sequence(tr) = [tr[state_addr(t)] for t=0:get_args(tr)[1]]

observation_retval_sequence(tr) = [tr[obs_addr(t)] for t=0:get_args(tr)[1]]

# TODO: once we have other types of trajectory generative functions,
# the actions may not be the argument -- so we will need to modify this
# function to check the type of the generative function and do the right thing.
action_sequence(tr) = get_args(tr)[2]
