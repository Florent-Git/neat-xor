package be.floshie.neat.ga.speciation;

import be.floshie.neat.ai.AdvancedNeuralNetwork;
import be.floshie.neat.ai.graph.Axon;
import be.floshie.neat.ga.Individual;
import io.vavr.Function1;
import io.vavr.Tuple;
import io.vavr.collection.HashMap;
import io.vavr.collection.List;
import io.vavr.collection.Map;
import lombok.RequiredArgsConstructor;
import lombok.val;


@RequiredArgsConstructor
public class SpeciationAlgorithm {
    private final double c1;
    private final double c2;
    private final double c3;
    private final double dt;

    public double getDistance(Individual i1, Individual i2) {
        val ann1 = i1.getAdvancedNeuralNetwork();
        val ann2 = i2.getAdvancedNeuralNetwork();

        val allAxons = List.ofAll(ann1.edgeSet()).appendAll(ann2.edgeSet());

        val matchingAxons = getMatchingAxons(ann1, ann2);
        val disjointAndExcessAxons = getDisjointAndExcessAxons(i1, i2);
        val disjointAxons = disjointAndExcessAxons.filter(axon -> axon._2 == AxonType.DISJOINT).keySet();
        val excessAxons = disjointAndExcessAxons.filter(axon -> axon._2 == AxonType.EXCESS).keySet();

        val N = allAxons.distinctBy(Axon::getInnovation).size();
        val E = excessAxons.size();
        val D = disjointAxons.size();

        val W = matchingAxons.groupBy(Axon::getInnovation)
            .map(axons -> axons._2.map(Axon::getWeight).toList())
            .map(weights -> Math.abs(weights.get(0) - weights.get(1)))
            .sum().doubleValue();

        return (c1 * E / N) + (c2 * D / N) + (c3 * W);
    }

    /**
     * Adjust the fitness of each individual by applying the sharing function.
     * Imagine having typealiases in Java...
     */
    private Function1<Map<Individual, Double>, Map<Individual, Double>> adjustFitnessFn = ((Function1<Map<Individual, Double>, Map<Individual, Double>>) individuals -> {
        return individuals.map((individual, fitness) -> {
            // For every other individual, we compute the distance, apply the sharing function and sum the results
            val sum = individuals
                .filter(it -> it._1 != individual)
                .map(it -> sh(getDistance(individual, it._1)))
                .sum()
                .doubleValue();

            return Tuple.of(individual, fitness / sum);
        });
    }).memoized();

    public Map<Individual, Double> adjustFitness(Map<Individual, Double> individuals) {
        return adjustFitnessFn.apply(individuals);
    }

    private int sh(double distance) {
        return distance < dt ? 1 : 0;
    }

    private List<Axon> getMatchingAxons(AdvancedNeuralNetwork ann1, AdvancedNeuralNetwork ann2) {
        List<Axon> matchingAxons = List.empty();
        for (val axon1 : ann1.edgeSet()) {
            for (val axon2 : ann2.edgeSet()) {
                if (axon1.getInnovation() == axon2.getInnovation()) {
                    matchingAxons = matchingAxons.append(axon1);
                    matchingAxons = matchingAxons.append(axon2);
                }
            }
        }
        return matchingAxons;
    }

    enum AxonType {
        DISJOINT,
        EXCESS
    }

    // To get the disjoint and excess genes, we need to find the axons that are not matching
    // Then, we need to know each individual's max innovation number
    // The excess genes are the axons with an innovation number higher than the max innovation number of the other individual
    // The disjoint genes are the axons with an innovation number that is lower than the minimum of the max innovation numbers of both individuals
    private Map<Axon, AxonType> getDisjointAndExcessAxons(Individual i1, Individual i2) {
        val ann1 = i1.getAdvancedNeuralNetwork();
        val ann2 = i2.getAdvancedNeuralNetwork();

        val maxInnovation1 = ann1.edgeSet().stream().map(Axon::getInnovation).max(Integer::compare).orElse(0);
        val maxInnovation2 = ann2.edgeSet().stream().map(Axon::getInnovation).max(Integer::compare).orElse(0);

        val minMaxInnovation = Math.min(maxInnovation1, maxInnovation2);
        val excessNeuralNetwork = maxInnovation1 > maxInnovation2 ? ann1 : ann2;

        val matchingAxons = getMatchingAxons(ann1, ann2);
        val nonMatchingAxons = List.ofAll(ann1.edgeSet()).appendAll(ann2.edgeSet()).removeAll(matchingAxons);

        val disjointAxons = nonMatchingAxons
            .filter(axon -> axon.getInnovation() <= minMaxInnovation)
            .collect(List.collector());

        val excessAxons = excessNeuralNetwork.edgeSet().stream()
            .filter(axon -> axon.getInnovation() > minMaxInnovation)
            .collect(List.collector());

        Map<Axon, AxonType> empty = HashMap.empty();

        return empty
            .merge(disjointAxons.toMap(axon -> axon, axon -> AxonType.DISJOINT))
            .merge(excessAxons.toMap(axon -> axon, axon -> AxonType.EXCESS));
    }
}
