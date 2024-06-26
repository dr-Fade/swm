using Lux

conv_1d_output_dims(n, k; depth=1, stride=1) =
    if depth == 1
        (n - k)÷stride + 1
    else
        (conv_1d_output_dims(n, k; depth=depth-1) - k)÷stride + 1
    end

conv_1d(n, k; depth=1, stride=1) = Lux.Chain(
    Lux.ReshapeLayer((n, 1)),
    [Lux.Conv((k,), 1=>1; stride=stride) for _ in 1:depth],
    Lux.FlattenLayer(),
), conv_1d_output_dims(n,k;depth=depth,stride=stride)
