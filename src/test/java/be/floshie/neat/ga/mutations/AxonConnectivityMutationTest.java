package be.floshie.neat.ga.mutations;

import be.floshie.neat.ai.AdvancedNeuralNetwork;
import be.floshie.neat.ga.Individual;
import lombok.val;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class AxonConnectivityMutationTest {
    private AdvancedNeuralNetwork startingNetwork = AdvancedNeuralNetwork.minimal(1, 1);
    private AdvancedNeuralNetwork mutatedNetwork;

    private Individual startingIndividual;
    private Individual mutatedIndividual;

    @BeforeEach
    void setUp() {
        startingIndividual = new Individual(startingNetwork);
        val axon = startingNetwork.edgeSet().stream().toList().get(0);
        startingNetwork.setEdgeWeight(axon, 0.5);
        mutatedIndividual = new AxonConnectivityMutation(1).mutate(startingIndividual);

        mutatedNetwork = mutatedIndividual.getAdvancedNeuralNetwork();
    }

    @Test
    void givenASimpleNeuronalNetwork_whenItMutatesAnEnabledConnectionWeight_thenTheWeightIsSetTo0() {
        val mutatedAxon = mutatedNetwork.edgeSet().stream().toList().get(0);
        assertEquals(0, mutatedNetwork.getEdgeWeight(mutatedAxon));
    }

    @Test
    void givenASimpleNeuronalNetwork_whenItMutatesADisabledConnectionWeight_thenTheWeightIsBackAtItsBeforeValue() {
        val axon = startingNetwork.edgeSet().stream().toList().get(0);
        startingNetwork.setEdgeWeight(axon, .5);
        val mutatedAxon = mutatedNetwork.edgeSet().stream().toList().get(0);
        mutatedNetwork.setEdgeWeight(mutatedAxon, .5);

        val reenabledIndividual = new AxonConnectivityMutation(1).mutate(mutatedIndividual);
        val reenabledNetwork = reenabledIndividual.getAdvancedNeuralNetwork();
        val reenabledAxon = reenabledNetwork.edgeSet().stream().toList().get(0);

        assertTrue(mutatedNetwork.getEdgeWeight(mutatedAxon) >= -1);
        assertTrue(mutatedNetwork.getEdgeWeight(mutatedAxon) <= 1);
        assertEquals(.5, reenabledNetwork.getEdgeWeight(reenabledAxon));
    }
}