package be.floshie.neat.ai.graph;

import io.vavr.Function1;
import lombok.*;

@Getter
@ToString
@EqualsAndHashCode
@RequiredArgsConstructor
public class Neuron {
    private final int id;
    private final Function1<Double, Double> activationFunction;
}
