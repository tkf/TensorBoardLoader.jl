using Documenter, TensorBoardLoader

makedocs(;
    modules=[TensorBoardLoader],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/tkf/TensorBoardLoader.jl/blob/{commit}{path}#L{line}",
    sitename="TensorBoardLoader.jl",
    authors="Takafumi Arakaki <aka.tkf@gmail.com>",
    assets=String[],
)

deploydocs(;
    repo="github.com/tkf/TensorBoardLoader.jl",
)
