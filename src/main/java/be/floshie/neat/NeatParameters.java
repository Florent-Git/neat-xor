package be.floshie.neat;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.dataformat.yaml.YAMLFactory;
import lombok.RequiredArgsConstructor;
import lombok.val;
import picocli.CommandLine;

import java.io.File;
import java.io.IOException;

@RequiredArgsConstructor
public class NeatParameters {
    public int populationSize = 100;
    public int inputSize = 2;
    public int outputSize = 1;
    public double weightMutationRate = 0.8;
    public double weightMutationNudgeMax = 0.5;
    public double weightMutationNudgeMin = -0.5;
    public double neuronMutationRate = 0.3;
    public double axonMutationRate = 0.1;
    public double axonMutationRateMaxWeight = 0.5;
    public double axonMutationRateMinWeight = -0.5;
    public double axonConnectivityMutationRate = 0.1;
    public double crossoverRate = 0.2;
    public int tournamentSize = 33;
    public int maxGenerations = 1000;

    /**
     * The coefficient for disjoint genes in the distance calculation
     */
    public double c1 = 1.0;

    /**
     * The coefficient for excess genes in the distance calculation
     */
    public double c2 = 1.0;

    /**
     * The coefficient for average weight difference in the distance calculation
     */
    public double c3 = 0.4;

    /**
     * The threshold for the distance calculation
     */
    public double dt = 0.1;

    public static NeatParameters load(File configFile) throws IOException {
        val mapper = new ObjectMapper(new YAMLFactory());
        return mapper.readValue(configFile, NeatParameters.class);
    }
}
