package be.floshie.neat;

import be.floshie.neat.ai.AdvancedNeuralNetwork;
import be.floshie.neat.ga.Individual;
import be.floshie.neat.ga.crossover.BasicCrossover;
import be.floshie.neat.ga.fitness.XORFitness;
import be.floshie.neat.ga.mutations.AxonConnectivityMutation;
import be.floshie.neat.ga.mutations.AxonMutation;
import be.floshie.neat.ga.mutations.NeuronMutation;
import be.floshie.neat.ga.mutations.WeightMutation;
import be.floshie.neat.ga.selection.TournamentSelection;
import be.floshie.neat.ga.speciation.SpeciationAlgorithm;
import be.floshie.neat.gui.TestSketch;
import io.vavr.Function1;
import io.vavr.Tuple2;
import io.vavr.collection.List;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import picocli.CommandLine;
import processing.core.PApplet;

import java.io.File;
import java.io.IOException;
import java.util.concurrent.Callable;
import java.util.stream.Stream;

@Slf4j
public class NeatMain implements Callable<Integer> {
    public static void main(String[] args) throws Exception {
        val neatMain = new NeatMain();
        val configFile = new File(NeatMain.class.getClassLoader().getResource("config.yml").getFile());
        neatMain.parameters = NeatParameters.load(configFile);
        val ret = neatMain.call();
        System.exit(ret);
    }

    private NeatParameters parameters;

    private List<Individual> individuals;

    @Override
    public Integer call() throws Exception {
        individuals = List.ofAll(Stream.generate(
            () -> new Individual(AdvancedNeuralNetwork.minimal(parameters.inputSize, parameters.outputSize))
        ).limit(parameters.populationSize));

        Function1<Individual, Individual> mutate = getMutationFn();

        val fitnessStrategy = new XORFitness(
            new SpeciationAlgorithm(
                parameters.c1,
                parameters.c2,
                parameters.c3,
                parameters.dt
            )
        );

        val tournamentSelection = new TournamentSelection(fitnessStrategy, parameters.tournamentSize);
        val crossoverStrategy = new BasicCrossover(parameters.crossoverRate, fitnessStrategy, tournamentSelection);

        int generation = 0;

        while (true) {
            val fitness = fitnessStrategy.getFitnesses(individuals);
            val bestIndividual = fitness.maxBy(Tuple2::_2).get()._1();

            var newIndividuals = List.of(bestIndividual);

            for (int i = 0; i < individuals.size() - 1; ++i) {
                val offspring = crossoverStrategy.crossover(individuals);
                val mutatedIndividual = mutate.apply(offspring);
                newIndividuals = newIndividuals.append(mutatedIndividual);
            }

            individuals = newIndividuals;

            log.info("Generation: {}", generation);
            log.info("Best fitness: {}", fitness.get(bestIndividual).get());
            log.info("Best individual: {}", bestIndividual);

            log.info("Best individual's performance:\n\t0, 0:\t{}\n\t0, 1:\t{}\n\t1, 0:\t{}\n\t1, 1:\t{}",
                bestIndividual.getAdvancedNeuralNetwork().feedForward(List.of(0.0, 0.0)).get(0),
                bestIndividual.getAdvancedNeuralNetwork().feedForward(List.of(0.0, 1.0)).get(0),
                bestIndividual.getAdvancedNeuralNetwork().feedForward(List.of(1.0, 0.0)).get(0),
                bestIndividual.getAdvancedNeuralNetwork().feedForward(List.of(1.0, 1.0)).get(0)
            );

            generation++;
        }
    }

    private Function1<Individual, Individual> getMutationFn() {
        val axonConnectivityMutation = new AxonConnectivityMutation(parameters.axonConnectivityMutationRate);
        val axonMutation = new AxonMutation(
            parameters.axonMutationRate,
            parameters.axonMutationRateMaxWeight,
            parameters.axonMutationRateMinWeight
        );
        val neuronMutation = new NeuronMutation(parameters.neuronMutationRate);
        val weightMutation = new WeightMutation(
            parameters.weightMutationRate,
            parameters.weightMutationNudgeMax,
            parameters.weightMutationNudgeMin
        );

        return Function1.of(axonConnectivityMutation::mutate)
            .andThen(axonMutation::mutate)
            .andThen(neuronMutation::mutate)
            .andThen(weightMutation::mutate);
    }
}
