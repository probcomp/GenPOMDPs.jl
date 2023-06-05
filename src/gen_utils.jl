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