package be.floshie.neat.ga.selection;

import be.floshie.neat.ga.Individual;
import be.floshie.neat.ga.fitness.IFitness;
import io.vavr.Function1;
import io.vavr.Tuple2;
import io.vavr.collection.List;
import lombok.RequiredArgsConstructor;

@RequiredArgsConstructor
public class TournamentSelection implements ISelection {
    private final IFitness fitnessFn;
    private final int tournamentSize;

    @Override
    public Individual select(List<Individual> individuals) {
        return individuals
            .zip(fitnessFn.getFitnesses(individuals).values())
            .shuffle()
            .subSequence(0, tournamentSize)
            .maxBy(Tuple2::_2)
            .map(Tuple2::_1)
            .get();
    }
}
