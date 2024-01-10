package be.floshie.neat.ai;

import io.vavr.Function1;
import lombok.AllArgsConstructor;

@AllArgsConstructor
public enum ActivationFunction implements Function1<Double, Double> {
    RELU(aDouble -> Math.max(0, aDouble)),
    LINEAR(aDouble -> aDouble),
    TANH(Math::tanh),
    SIGMOID(aDouble -> 1 / (1 + Math.exp(-aDouble))),
    ;

    private final Function1<Double, Double> function;

    @Override
    public Double apply(Double aDouble) {
        return function.apply(aDouble);
    }
}
