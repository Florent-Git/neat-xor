package be.floshie.neat.ga.mutations;

import be.floshie.neat.ga.Individual;
import lombok.RequiredArgsConstructor;
import lombok.val;

@RequiredArgsConstructor
public class WeightMutation implements MutationStrategy {
    private final double mutationRate;
    private final double nudgeMax;
    private final double nudgeMin;

    @Override
    public Individual mutate(Individual individual) {
        if (Math.random() > mutationRate) {
            return individual;
        }

        val network = individual.getAdvancedNeuralNetwork();
        // Create a new network
        val copiedNetwork = network.copy();

        // Choose a random axon to mutate
        val randomAxon = copiedNetwork.edgeSet()
            .stream()
            .toList()
            .get((int) (Math.random() * copiedNetwork.edgeSet().size()));

        val ancientWeight = network.getEdgeWeight(randomAxon);

        // Mutate the weight
        copiedNetwork.setEdgeWeight(randomAxon, ancientWeight + (Math.random() * (nudgeMax - nudgeMin) + nudgeMin));

        return individual.withAdvancedNeuralNetwork(copiedNetwork);
    }
}
