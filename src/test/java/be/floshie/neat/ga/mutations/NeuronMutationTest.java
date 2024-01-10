package be.floshie.neat.ga.mutations;

import be.floshie.neat.ai.AdvancedNeuralNetwork;
import be.floshie.neat.ga.Individual;
import lombok.val;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class NeuronMutationTest {
    @Test
    void givenASimpleNeuronNetwork_whenMutating_thenTheNetworkHasOneMoreNeuron() {
        val network = AdvancedNeuralNetwork.minimal(1, 1);
        val fullMutation = new NeuronMutation(1);

        val newNetwork = fullMutation
                .mutate(new Individual(network))
                .getAdvancedNeuralNetwork();

        assertEquals(3, newNetwork.vertexSet().size());
    }
}