module TensorBoardLoader

using Transducers
using Transducers: Eduction, @next, complete
using TensorBoardLogger
using Tables

include("dynamicrow.jl")

struct EventLoader
    path::String
end

mutable struct EventStream
    io::IO
    paths::Vector{String}
end

function Base.open(el::EventLoader)
    paths = logfilepaths(el.path)
    path1 = popfirst!(paths)
    return EventStream(open(path1), paths)
end

Base.close(es::EventStream) = close(es.io)

islogfile(path) = occursin("tfevents", path)
function logfiletime(path)
    m = match(r"tfevents\.([0-9]+(\.[0-9]+(e[0-9]+)?)?)", path)
    return m === nothing ? NaN : m : parse(Float64, m.captures[1])
end

function logfilepaths(path)
    isfile(path) && return [path]
    # TOOD: better way to sort it
    return joinpath.(
        path,
        sort!(filter(islogfile, readdir(path)); by=logfiletime),
    )
end

"""
    read_event(f::IOStream) :: TensorBoardLogger.Event

https://github.com/PhilipVinc/TensorBoardLogger.jl/issues/38#issuecomment-508704329
"""
function read_event(f::IOStream)
    header = read(f, 8)
    crc_header = read(f, 4)

    # check
    crc_header_ck = reinterpret(UInt8, UInt32[TensorBoardLogger.masked_crc32c(header)])
    @assert crc_header == crc_header_ck

    # DATA
    data_len = first(reinterpret(Int64, header))
    data = read(f, data_len)
    crc_data = read(f, 4)

    # check
    crc_data_ck = reinterpret(UInt8, UInt32[TensorBoardLogger.masked_crc32c(data)])
    @assert crc_data == crc_data_ck

    pb = PipeBuffer(data)
    ev = TensorBoardLogger.readproto(pb, TensorBoardLogger.Event())
    return ev
end

# It makes sense to have better `EventStream` and `EventLoader` but to
# be friendly to iterator-based ecosystem:
function Base.iterate(es::EventStream, _=nothing)
    eof(es.io) || return (read_event(es.io), nothing)
    isempty(es.paths) && return nothing
    close(es.io)
    es.io = open(popfirst!(es.paths))
    return iterate(es)
end

Transducers.__foldl__(rf, acc, el::EventLoader) =
    open(el) do es
        Transducers.__foldl__(rf, acc, es)
    end

"""
    has_summary_value(e::TensorBoardLogger.Event) :: Bool
"""
has_summary_value(e) = isdefined(e, :summary) && isdefined(e.summary, :value)

"""
    has_non_scalar(s::TensorBoardLogger.Summary_Value) :: Bool
"""
has_non_scalar(s) =
    isdefined(s, :obsolete_old_style_histogram) ||
    isdefined(s, :image) ||
    isdefined(s, :histo) ||
    isdefined(s, :audio) ||
    isdefined(s, :tensor)

only_scalar() = Filter(!has_non_scalar)

cat_summary() =
    MapCat() do e
        if has_summary_value(e)
            e.summary.value
        else
            TensorBoardLogger.Summary_Value[]
        end
    end

summaries_as_dict(xf_summary = Map(identity)) =
    Filter(has_summary_value) |>
    PartitionBy(e -> e.step) |>
    Map() do events
        xf = cat_summary() |> xf_summary |> Map() do e
            e.tag => e
        end
        dict = Dict{String, TensorBoardLogger.Summary_Value}()
        dict = foldl(push!, xf, events, init=dict)
        (events[1].step, dict)
    end

scalars_as_dict() =
    Filter(has_summary_value) |>
    PartitionBy(e -> e.step) |>
    Map() do events
        xf = cat_summary() |> only_scalar() |> Map() do e
            e.tag => e.simple_value
        end
        dict = foldl(push!, xf, events, init=Dict{String, Float32}())
        (events[1].step, dict)
    end

scalars_as_row(key = "step") =
    scalars_as_dict() |> include_step(key) |> Map(dynamicrow)

include_step(key = "step") = Map() do (step, dict)
    dict[key] = step
    dict
end

"""
    loadscalars(f, path)

Open tensorboard log file at `path`, create an table-like lazy iterator,
evaluate single-argument function `f` on it, and then close the file.

The iterator passed to `f` satisfies `Tables.rows` interface.

# Examples
```julia
loadscalars(DataFrame, "logdir")
loadscalars(DataFrame, "logdir/events.out.tfevents....")
```
"""
loadscalars(f, path::AbstractString) = loadscalars(f, EventLoader(path))
loadscalars(f, el::EventLoader) =
    open(el) do es
        f(eduction(scalars_as_row(), es))
    end

# Find a better implementation and move this to Transducers.jl
Tables.rows(table::Eduction) = table
Tables.rowaccess(::Type{<:Eduction}) = true
Tables.istable(::Type{<:Eduction}) = true

end # module
