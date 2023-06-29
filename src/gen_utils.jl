# Construct a tuple -- in a way that is compatible with Gen's
# change-hint ("Diff") system for incremental computation.
function maketuple(args...)
    if any(arg isa Diffed for arg in args)
        return Diffed(
            map(strip_diff, args),
            all(get_diff(arg) == NoChange() for arg in args) ? NoChange() : UnknownChange()
        )
    end
    return args
end

### Utilities for manipulating Gen addresses ###
nest_at(prefix, suffix) = prefix => suffix
nest_at(prefix::Pair, suffix) = prefix.first => nest_at(prefix.second, suffix)
function nest_choicemap(to_nest, addr)
    cm = choicemap()
    Gen.set_submap!(cm, addr, to_nest)
    return cm
end

# I have made this PR [https://github.com/probcomp/Gen.jl/pull/509/commits/ebeee8ee8da1bfdc022e9350e63997722b279922]
# to Gen.jl to add this function:
function Base.copy(state::Gen.ParticleFilterState{U}) where U
    Gen.ParticleFilterState{U}(
        copy(state.traces),
        copy(state.new_traces),
        copy(state.log_weights),
        state.log_ml_est,
        copy(state.parents)
    )
end