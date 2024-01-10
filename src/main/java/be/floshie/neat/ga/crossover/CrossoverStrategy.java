package be.floshie.neat.ga.crossover;

import be.floshie.neat.ga.Individual;
import io.vavr.collection.List;

public interface CrossoverStrategy {
    Individual crossover(List<Individual> individuals);
}
