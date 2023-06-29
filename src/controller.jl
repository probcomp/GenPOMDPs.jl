struct Controller
    controller :: Gen.GenerativeFunction # (controller state, obs) -> (action, next controller state)
    init_state :: Any                    # Initial controller state
end