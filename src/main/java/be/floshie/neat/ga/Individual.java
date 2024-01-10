package be.floshie.neat.ga;

import be.floshie.neat.ai.AdvancedNeuralNetwork;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.With;

@Getter
@RequiredArgsConstructor
public class Individual {
    @With
    private final AdvancedNeuralNetwork advancedNeuralNetwork;
}
