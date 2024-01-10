package be.floshie.neat.ga.mutations;

import be.floshie.neat.ga.Individual;
import io.vavr.collection.List;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.jgrapht.graph.GraphCycleProhibitedException;

import java.util.Objects;

@Slf4j
@RequiredArgsConstructor
public class AxonMutation implements MutationStrategy {
    private final double mutationRate;
    private final double maxWeight;
    private final double minWeight;

    /**
     * Mutates an individual by adding a new axon between two neurons. The weight of the axon is randomly generated
     * between the min and max weight. The mutation rate determines the probability of this mutation happening.
     *
     * @param individual The individual to mutate
     * @return The mutated individual
     */
    @Override
    public Individual mutate(Individual individual) {
        if (Math.random() > mutationRate) {
            return individual;
        }

        val network = individual.getAdvancedNeuralNetwork();
        val copiedNetwork = network.copy();

        val neurons = List.ofAll(copiedNetwork.vertexSet());

        val possibleAxons = neurons.crossProduct()
            .toMap(t -> t, tuple -> copiedNetwork.getEdge(tuple._1(), tuple._2()))
            .filterValues(Objects::isNull) // Filter out existing axons
            .filterKeys(tuple -> !tuple._1().equals(tuple._2())) // Filter out self-connections
            .filterKeys(tuple -> copiedNetwork.inDegreeOf(tuple._2()) != 0) // Filter out input neurons
            .filterKeys(tuple -> copiedNetwork.outDegreeOf(tuple._1()) != 0) // Filter out output neurons
            .filterKeys(tuple -> copiedNetwork.getEdge(tuple._2(), tuple._1()) == null); // Filter out existing reverse axons

        if (possibleAxons.isEmpty()) {
            return individual;
        }

        val axonToMutate = possibleAxons.keySet()
            .toList()
            .get((int) (Math.random() * possibleAxons.size()));

        try {
            copiedNetwork.addAxon(
                    axonToMutate._1(),
                    axonToMutate._2(),
                    Math.random() * (maxWeight - minWeight) + minWeight
            );
        } catch (GraphCycleProhibitedException e) {
//            log.warn("Ignoring axon mutation because it would create a cycle");
        }

        return individual.withAdvancedNeuralNetwork(copiedNetwork);
    }
}
