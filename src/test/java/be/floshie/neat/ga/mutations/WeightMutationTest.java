package be.floshie.neat.ga.mutations;

import be.floshie.neat.ai.AdvancedNeuralNetwork;
import be.floshie.neat.ga.Individual;
import lombok.val;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class WeightMutationTest {
    @Test
    void givenANeuralNetwork_whenMutatingWeights_thenWeightsAreMutated() {
        val neuralNetwork = AdvancedNeuralNetwork.minimal(1, 1);

        val weightMutation = new WeightMutation(1, .5, -.5);
        val mutatedNetwork = weightMutation
            .mutate(new Individual(neuralNetwork))
            .getAdvancedNeuralNetwork();

        val currentWeights = neuralNetwork
            .edgeSet()
            .stream()
            .map(neuralNetwork::getEdgeWeight)
            .sorted()
            .toList();

        val mutatedWeights = mutatedNetwork
            .edgeSet()
            .stream()
            .map(mutatedNetwork::getEdgeWeight)
            .sorted()
            .toList();

        assertNotEquals(currentWeights, mutatedWeights);
    }
}