using Lux

conv_1d_output_dims(n, k; depth=1, stride=1) =
    if depth == 1
        (n - k)÷stride + 1
    else
        current_out = (n - k)÷stride + 1
        (conv_1d_output_dims(current_out, k; depth=depth-1))÷stride 
    end

conv_1d(input_n, kernel_n; depth=1, stride=1, activation=identity) = Lux.Chain(
    Lux.ReshapeLayer((input_n, 1)),
    [Lux.Conv((kernel_n,), 1=>1, activation; stride=stride) for _ in 1:depth],
    Lux.FlattenLayer(),
), conv_1d_output_dims(input_n, kernel_n; depth=depth, stride=stride)
