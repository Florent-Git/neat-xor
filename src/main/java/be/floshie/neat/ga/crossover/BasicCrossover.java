package be.floshie.neat.ga.crossover;

import be.floshie.neat.ai.AdvancedNeuralNetwork;
import be.floshie.neat.ai.graph.Axon;
import be.floshie.neat.ai.graph.Neuron;
import be.floshie.neat.ga.Individual;
import be.floshie.neat.ga.fitness.IFitness;
import be.floshie.neat.ga.selection.ISelection;
import io.vavr.Function1;
import io.vavr.collection.HashSet;
import io.vavr.collection.List;
import io.vavr.collection.Set;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import lombok.val;

import java.util.function.Consumer;
import java.util.function.Supplier;

@Slf4j
@RequiredArgsConstructor
public class BasicCrossover implements CrossoverStrategy {
    private final double crossoverRate;
    private final IFitness fitnessFn;
    private final ISelection selection;

    @Override
    public Individual crossover(List<Individual> individuals) {
        val parent1 = selection.select(individuals);
        val parent2 = selection.select(individuals);

        val fitness1 = fitnessFn.getFitnesses(individuals).get(parent1).get();
        val fitness2 = fitnessFn.getFitnesses(individuals).get(parent2).get();

        val ann = AdvancedNeuralNetwork.empty();

        val neurons1 = getNeurons(parent1);
        val neurons2 = getNeurons(parent2);

        val neuronUnion = neurons1.union(neurons2);
        neuronUnion.forEach(ann::addVertex);

        val best = fitness1 > fitness2 ? parent1 : parent2;

        val innovationNumbers1 = getInnovationNumbers(parent1);
        val innovationNumbers2 = getInnovationNumbers(parent2);

        // We keep the innovation numbers that are in both parents
        // For each innovation number, we randomly pick one of the two parents' axons
        val matchingInnovationNumbers = innovationNumbers1.intersect(innovationNumbers2);
        matchingInnovationNumbers.forEach(addEdge(ann, () -> Math.random() < crossoverRate ? parent1 : parent2));

        // We keep the remaining innovation numbers that are only in the best parent
        val remainingInnovationNumbers =
            best == parent1
                ? innovationNumbers1.removeAll(matchingInnovationNumbers)
                : innovationNumbers2.removeAll(matchingInnovationNumbers);

        remainingInnovationNumbers.forEach(addEdge(ann, () -> best));

        // Get the max neuron id
        val maxNeuronId = neuronUnion.map(Neuron::getId).max().get();

        // Get the max axon id
        val maxAxonId = ann.edgeSet().stream().map(Axon::getInnovation).max(Integer::compareTo).get();

        // Set the neuron and axon ids
        ann.setAxonId(maxAxonId + 1);
        ann.setNeuronId(maxNeuronId + 1);

        return new Individual(ann);
    }

    private Consumer<Integer> addEdge(AdvancedNeuralNetwork ann, Supplier<Individual> individualSupplier) {
        return innovationNumber -> {
            val individual = individualSupplier.get();

            val axon = individual.getAdvancedNeuralNetwork().getEdge(innovationNumber);
            val source = individual.getAdvancedNeuralNetwork().getEdgeSource(axon);
            val target = individual.getAdvancedNeuralNetwork().getEdgeTarget(axon);

            ann.addEdge(source, target, axon);
        };
    }

    private Set<Integer> getInnovationNumbers(Individual individual) {
        return individual
            .getAdvancedNeuralNetwork()
            .edgeSet()
            .stream()
            .map(Axon::getInnovation)
            .sorted()
            .collect(HashSet.collector());
    }

    private Set<Neuron> getNeurons(Individual individual) {
        return individual
            .getAdvancedNeuralNetwork()
            .vertexSet()
            .stream()
            .collect(HashSet.collector());
    }
}
