package be.floshie.neat.ga.fitness;

import be.floshie.neat.ga.Individual;
import be.floshie.neat.ga.speciation.SpeciationAlgorithm;
import io.vavr.Function1;
import io.vavr.collection.List;
import io.vavr.collection.Map;
import lombok.RequiredArgsConstructor;
import lombok.val;

@RequiredArgsConstructor
public class XORFitness implements IFitness {
    private final SpeciationAlgorithm speciationAlgorithm;

    @Override
    public Map<Individual, Double> getFitnesses(List<Individual> individuals) {
        val fitnesses = individuals.toMap(it -> it, this::getBasicFitness);
        return speciationAlgorithm.adjustFitness(fitnesses);
    }

    private double getBasicFitness(Individual in) {
        Function1<Individual, Double> fn = individual -> {
            List<List<Double>> inputs = List.of(
                List.of(0.0, 0.0),
                List.of(0.0, 1.0),
                List.of(1.0, 0.0),
                List.of(1.0, 1.0)
            );

            List<Double> expectedOutputs = List.of(
                0.0,
                1.0,
                1.0,
                0.0
            );

            val actualOutputs = inputs.map(individual.getAdvancedNeuralNetwork()::feedForward)
                .map(List::head);

            double sumSquaredError = 0.0;
            for (int i = 0; i < expectedOutputs.size(); i++) {
                double error = expectedOutputs.get(i) - actualOutputs.get(i);
                sumSquaredError += Math.pow(error, 2);
            }

            val mse = sumSquaredError / expectedOutputs.size();
            return 1.0 / (1.0 + mse);
        };

        return Function1.of(fn).memoized().apply(in);
    }
}
