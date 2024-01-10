package be.floshie.neat.ga.mutations;

import be.floshie.neat.ai.AdvancedNeuralNetwork;
import be.floshie.neat.ga.Individual;

public interface MutationStrategy {
    Individual mutate(Individual individual);
}
