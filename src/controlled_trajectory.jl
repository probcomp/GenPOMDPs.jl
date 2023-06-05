using Suppressor: @suppress

"""
The probability distribution over the trajectories resulting from applying
a fixed action sequence in a POMDP.

This is represented as a generative function accepting inputs
(T, action_sequence, params), where length(action_sequence) ≥ T.
`T` controls the length of the trajectory, and
`params` can control the details of the POMDP model.  (So,
more precisely, the generative function is a family of distributions
over trajectories, indexed by `T` and `params`.)

The generative function returns a pair (init, steps), where:
- init is a pair (state0, obs0)
- steps is a vector of (state_t, obs_t) pairs, for each 1 ≤ t ≤ T.

The choicemap of this generative function will look like:
- :init
    - :state -> choicemap of p.init
    - :obs -> choicemap of p.obs
- :steps
    - 1
        - :state -> choicemap of p.step
        - :obs -> choicemap of p.obs
    - 2
        - :state -> choicemap of p.step
        - :obs -> choicemap of p.obs
    - ...
    - T
        - :state -> choicemap of p.step
        - :obs -> choicemap of p.obs
"""
function ControlledTrajectoryModel(p::GenPOMDP)

    # We need to use @eval in order for these generative functions to be
    # evaluated at the top-level scope, which is necessary for them to be
    # compiled properly.
    # `model` will evaluate to the _ControlledTrajectoryModel generative function.
    model = @eval begin
        @gen (static, diffs) function _init(params)
            state ~ ($p.init)(params)
            obs ~ ($p.obs)(state, params)
            return maketuple(state, obs)
        end
        @gen (static, diffs) function _step(t, prev, actions, params)
            (prev_state, prev_obs) = prev

            state ~ ($p.step)(prev_state, actions[t], params)
            obs ~ ($p.obs)(state, params)
            return maketuple(state, obs)
        end

        @gen (static, diffs) function _ControlledTrajectoryModel(T, action_sequence, params)
            init ~ _init(params)
            steps ~ Unfold(_step)(T, init, action_sequence, params)

            # Return:
            # init - (state0, obs0)
            # steps - vector of (state_t, obs_t), for each t ≥ 1
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
# function state_sequence(tr)
#     state0 = get_retval(tr)[1][1]
#     states_rest = [state for (state, obs) in get_retval(tr)[2]]
#     return vcat([state0], states_rest)
# end
# function obs_sequence(tr)
#     obs0 = get_retval(tr)[1][2]
#     obss_rest = [obs for (state, obs) in get_retval(tr)[2]]
#     return vcat([obs0], obss_rest)
# end

"""
Used for interactive simulation from a POMDP environment,
in which actions are chosen externally.

Returns a pair (tr, onaction), where
- tr is an observable of a trajectory in the environment
- onaction is a function which accepts an action and extends
    the trajectory by taking that action and simulating from the
    dynamics and observation model.

Example:
```
    tr, onaction = interactive_world_trace(trajectory_model, params)

    # print the length of the trajectory
    println(1 + length(get_retval(tr[]))) # Prints: 1

    onaction(:left)
    onaction(:left)

    # print the length of the trajectory
    println(1 + length(get_retval(tr[]))) # Prints: 3

    on(tr) do tr
        println("The trajectory has changed!")
    end

    onaction(:right) # Prints: "The trajectory has changed!"
    println(1 + length(get_retval(tr[]))) # Prints: 4
```
"""
function interactive_world_trace(trajectory_model, params)
    tr = simulate(trajectory_model, (0, [], params))
    tr = Observables.Observable(tr)

    function onaction(action)
        past_T, past_actions = get_args(tr[])[1:2]

        # Update the trace by simulating from the model
        # with the new action.
        newtr = Gen.update(
            tr[],

            # Add the new action to the action sequence
            (past_T + 1, vcat(past_actions, [action]), params),

            (
                Gen.IntDiff(1),

                # Say that the action argument has not changed.
                # (In this model structure, the correct update
                # will occur if we tell Gen this, since only
                # the last action has changed.
                # This will help performance.
                # Eventually we need to add a better way for Gen
                # to achieve this performance gain -- ie. we need
                # a specific type of change hint which captures the fact
                # that only the last action has changed.)
                NoChange(),

                NoChange() # Params have not changed
            ),

            # No constraints for the update -- simulate freely from the model
            EmptyChoiceMap()
        )[1]

        # [x[] = ... syntax
        # updates the observable x and notifies subscribers.]
        tr[] = newtr
    end


    return tr, onaction
end