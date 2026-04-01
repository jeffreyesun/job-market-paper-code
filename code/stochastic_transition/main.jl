
using CUDA, cuDNN
using Flux
using ParameterSchedulers: Stateful, CosAnneal, next!

include("../transition/main.jl")

include("neural_nets.jl")
include("simulate.jl")
include("train.jl")
