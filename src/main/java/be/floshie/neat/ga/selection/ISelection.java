package be.floshie.neat.ga.selection;

import be.floshie.neat.ga.Individual;
import io.vavr.collection.List;

public interface ISelection {
    Individual select(List<Individual> individuals);
}
