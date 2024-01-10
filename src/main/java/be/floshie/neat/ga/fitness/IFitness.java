package be.floshie.neat.ga.fitness;

import be.floshie.neat.ga.Individual;
import io.vavr.collection.List;
import io.vavr.collection.Map;

public interface IFitness {
    Map<Individual, Double> getFitnesses(List<Individual> individuals);
}
