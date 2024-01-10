package be.floshie.neat.ga.mutations;

import be.floshie.neat.ai.ActivationFunction;
import be.floshie.neat.ai.AdvancedNeuralNetwork;
import be.floshie.neat.ga.Individual;
import lombok.RequiredArgsConstructor;
import lombok.val;

@RequiredArgsConstructor
public class NeuronMutation implements MutationStrategy {
    private final double mutationRate;

    /**
     * Mutates the individual by adding a neuron between two existing neurons with an existing axon. The weight of the
     * axon from the source to the new neuron is 1, the weight of the axon from the new neuron to the target is the
     * weight of the axon that was split.
     * @param individual
     * @return
     */
    @Override
    public Individual mutate(Individual individual) {
        if (Math.random() > mutationRate)
            return individual;

        val network = individual.getAdvancedNeuralNetwork();
        val copiedNetwork = network.copy();

        // Get a random axon
        val randomAxon = copiedNetwork
                .edgeSet()
                .stream()
                .toList()
                .get((int) (Math.random() * copiedNetwork.edgeSet().size()));

        // Split the axon
        val source = copiedNetwork.getEdgeSource(randomAxon);
        val target = copiedNetwork.getEdgeTarget(randomAxon);

        copiedNetwork.removeEdge(randomAxon);

        val newNeuron = copiedNetwork.addNeuron(ActivationFunction.SIGMOID);
        copiedNetwork.addAxon(source, newNeuron, 1);
        copiedNetwork.addAxon(newNeuron, target, network.getEdgeWeight(randomAxon));

        return individual.withAdvancedNeuralNetwork(copiedNetwork);
    }
}
