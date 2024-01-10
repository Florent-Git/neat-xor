package be.floshie.neat.ga.mutations;

import be.floshie.neat.ai.ActivationFunction;
import be.floshie.neat.ai.AdvancedNeuralNetwork;
import be.floshie.neat.ai.graph.Axon;
import be.floshie.neat.ai.graph.Neuron;
import be.floshie.neat.ga.Individual;
import lombok.val;
import org.jgrapht.graph.DirectedAcyclicGraph;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class AxonMutationTest {
    @Test
    void givenASimpleDenseNeuralNetwork_whenMutatingAxon_thenNothingHappens() {
        val neuralNetwork = AdvancedNeuralNetwork.minimal(1, 1);
        val axonMutation = new AxonMutation(1, 0, 1);

        val mutatedNeuralNetwork = axonMutation
            .mutate(new Individual(neuralNetwork))
            .getAdvancedNeuralNetwork();

        assertEquals(neuralNetwork.edgeSet(), mutatedNeuralNetwork.edgeSet());
    }

    @Test
    void givenANeuralNetwork_whenMutatingAxon_thenAnUnexistingAxonIsCreated() {
        val neuralNetworkGraph = new DirectedAcyclicGraph<Neuron, Axon>(
            null,
            () -> new Axon(-1),
            true
        );

        val neuralNetwork = AdvancedNeuralNetwork.from(neuralNetworkGraph);

        val n0 = neuralNetwork.addNeuron(ActivationFunction.LINEAR); // Input
        val n1 = neuralNetwork.addNeuron(ActivationFunction.LINEAR);
        val n2 = neuralNetwork.addNeuron(ActivationFunction.LINEAR);
        val n3 = neuralNetwork.addNeuron(ActivationFunction.LINEAR); // Output

        neuralNetwork.addAxon(n0, n1, 1);
        neuralNetwork.addAxon(n1, n2, 1);
        neuralNetwork.addAxon(n2, n3, 1);

        val axonMutation = new AxonMutation(1, 0, 1);

        val mutatedNeuralNetwork = axonMutation
            .mutate(new Individual(neuralNetwork))
            .getAdvancedNeuralNetwork();
        val edgeSetDifference = mutatedNeuralNetwork.edgeSet().stream()
                .filter(axon -> !neuralNetwork.edgeSet().contains(axon))
                .toList();

        val edge = edgeSetDifference.get(0);
        val source = mutatedNeuralNetwork.getEdgeSource(edge);
        val target = mutatedNeuralNetwork.getEdgeTarget(edge);

        val valid = (source.equals(n0) && target.equals(n3))
                || (source.equals(n1) && target.equals(n3))
                || (source.equals(n0) && target.equals(n2));

        assertEquals(1, edgeSetDifference.size());
        assertTrue(valid);
    }
}