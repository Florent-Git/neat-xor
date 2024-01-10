package be.floshie.neat.ga.mutations;

import be.floshie.neat.ga.Individual;
import lombok.RequiredArgsConstructor;
import lombok.val;

@RequiredArgsConstructor
public class AxonConnectivityMutation implements MutationStrategy {
    private final double mutationRate;

    /**
     * Mutate the individual by enabling or disabling a random axon.
     * The process is done by randomly selecting an axon and then toggling its enabled state.
     * If the axon was enabled, it is disabled and vice versa.
     * @param individual The individual to mutate
     * @return The mutated individual
     */
    @Override
    public Individual mutate(Individual individual) {
        if (Math.random() > mutationRate)
            return individual;

        val network = individual.getAdvancedNeuralNetwork();
        val newNetwork = individual.getAdvancedNeuralNetwork().copy();

        val randomEdge = newNetwork.edgeSet()
            .stream()
            .toList()
            .get((int) (Math.random() * newNetwork.edgeSet().size()));

        val edgeSource = network.getEdgeSource(randomEdge);
        val edgeTarget = network.getEdgeTarget(randomEdge);
        val edgeWeight = network.getEnabledEdgeWeight(randomEdge);

        newNetwork.removeEdge(randomEdge);
        newNetwork.addEdge(
            edgeSource,
            edgeTarget,
            randomEdge.withEnabled(!randomEdge.isEnabled())
        );

        newNetwork.setEdgeWeight(randomEdge, edgeWeight);

        return individual.withAdvancedNeuralNetwork(newNetwork);
    }
}
