package be.floshie.neat.ai;

import be.floshie.neat.ai.graph.Axon;
import be.floshie.neat.ai.graph.NeuralGraph;
import be.floshie.neat.ai.graph.Neuron;
import io.vavr.collection.HashMap;
import io.vavr.collection.List;
import io.vavr.collection.Map;
import lombok.*;
import org.jgrapht.graph.DirectedAcyclicGraph;

import java.util.function.Supplier;

/**
 * An advanced neural network implementation using JGraphT's {@link org.jgrapht.graph.DirectedAcyclicGraph}.
 * This specific graph implementation is used because it allows for a topological order iterator
 * that is refreshed every time an edge is modified.
 */
public class AdvancedNeuralNetwork implements NeuralGraph {
    @lombok.experimental.Delegate(types = NeuralGraph.class)
    private final DirectedAcyclicGraph<Neuron, Axon> graph;

    private Map<Integer, Double> activationValues;

    @Setter
    private int axonId = 0;

    @Setter
    private int neuronId = 0;

    /**
     * Create a new neural network from a graph
     * @param graph The graph to create the neural network from
     */
    private AdvancedNeuralNetwork(
        DirectedAcyclicGraph<Neuron, Axon> graph
    ) {
        this.graph = graph;
        this.activationValues = HashMap.empty();
    }

    /**
     * Add an axon to the neural network with a weight. The axon will be enabled by default.
     * @param source The source neuron
     * @param target The target neuron
     * @param weight The weight of the axon
     * @return The axon that was added
     */
    public Axon addAxon(Neuron source, Neuron target, double weight) {
        val axon = new Axon(axonId++);
        addEdge(source, target, axon);
        setEdgeWeight(axon, weight);
        return axon;
    }

    /**
     * Add a neuron to the neural network with an activation function.
     * @param activationFunction The activation function of the neuron
     * @return The neuron that was added
     */
    public Neuron addNeuron(ActivationFunction activationFunction) {
        val neuron = new Neuron(neuronId++, activationFunction);
        addVertex(neuron);
        return neuron;
    }

    @Override
    public void setEdgeWeight(Axon axon, double weight) {
        axon.setWeight(weight);
        graph.setEdgeWeight(axon, weight);
    }

    /**
     * Get the weight of an axon. If the axon is disabled, the weight will be 0.
     * @param axon The axon to get the weight of
     * @return The weight of the axon
     */
    @Override
    public double getEdgeWeight(Axon axon) {
        return axon.isEnabled() ? graph.getEdgeWeight(axon) : 0;
    }

    /**
     * Get the weight of an axon regardless of whether it is enabled or not.
     * @param axon The axon to get the weight of
     * @return The weight of the axon
     */
    public double getEnabledEdgeWeight(Axon axon) {
        return graph.getEdgeWeight(axon);
    }

    /**
     * Get the axon with the given innovation number
     * @param innovationNumber The innovation number of the axon
     * @return The axon with the given innovation number
     */
    @SneakyThrows
    public Axon getEdge(int innovationNumber) {
        return List.ofAll(edgeSet())
            .find(axon -> axon.getInnovation() == innovationNumber)
            .getOrElseThrow(getError(ErrorType.AXON_NOT_FOUND));
    }

    /**
     * Feed forward the input values through the neural network. The input values are linked to the input neurons
     * by their index in the list.
     * @param inputValues The input values
     * @return The output values
     */
    public List<Double> feedForward(List<Double> inputValues) {
        return this.feedForward(
            List.range(0, inputValues.size())
                .toMap(i -> Map.entry(i + 1, inputValues.get(i))));
    }

    /**
     * Feed forward the input values through the neural network
     *
     * @param inputValues The input values with the neuron id as key and the value as value
     * @return The output values
     */
    @SneakyThrows
    protected List<Double> feedForward(Map<Integer, Double> inputValues) {
        val inputIterator = graph.iterator();

        activationValues = activationValues.put(0, 1.0); // Bias neuron
        // Put the input values in the activation values
        for (val input : inputValues) {
            activationValues = activationValues.put(input._1(), input._2());
        }

        while (inputIterator.hasNext()) {
            val neuron = inputIterator.next();

            if (activationValues.containsKey(neuron.getId()))
                continue;

            // Get all incoming activations
            val incomingActivation = List.ofAll(incomingEdgesOf(neuron))
                .map(axon -> {
                    val source = getEdgeSource(axon);
                    val incomingValue = activationValues.get(source.getId());
                    val axonWeight = getEdgeWeight(axon);
                    return incomingValue.get() * (axonWeight == Double.MIN_VALUE ? 0 : axonWeight);
                })
                .sum()
                .doubleValue();

            val activatedValue = neuron.getActivationFunction().apply(incomingActivation);
            activationValues = activationValues.put(neuron.getId(), activatedValue);
        }

        // Get output neurons
        val outputNeurons = List.ofAll(vertexSet())
            .filter(neuron -> outDegreeOf(neuron) == 0);

        val output = outputNeurons.map(neuron -> {
            try {
                return activationValues.get(neuron.getId())
                    .getOrElseThrow(getError(ErrorType.BAD_INPUT));
            } catch (Throwable e) {
                throw new RuntimeException(e);
            }
        });

        activationValues = HashMap.empty();
        return output;
    }

    /**
     * Enum for the different types of errors that can occur
     */
    private enum ErrorType {
        BAD_INPUT,
        AXON_NOT_FOUND
    }

    /**
     * Get the error for the given error type
     * @param errorType The error type
     * @return The error
     */
    private static Supplier<Throwable> getError(ErrorType errorType) {
        return () -> switch (errorType) {
            case BAD_INPUT ->
                new IllegalArgumentException("Input values are not valid. Verify that the input values are valid and correctly linked to an input.");
            case AXON_NOT_FOUND -> new IllegalArgumentException("Axon not found");
        };
    }

    /**
     * Create a minimal neural network with the given input and output size with a linear activation function. The
     * input and output neurons will have a bias neuron. The axons will have a random weight between -1 and 1 and will
     * be enabled by default. The axons will be fully connected.
     * @param input The input size
     * @param output The output size
     * @return The minimal neural network
     */
    public static AdvancedNeuralNetwork minimal(int input, int output) {
        val graph = new DirectedAcyclicGraph<Neuron, Axon>(
            null,
            () -> new Axon(-1),
            true
        );

        val ann = AdvancedNeuralNetwork.from(graph);

        val inputNeurons = List.range(0, input + 2) // + 2 for bias
            .map(i -> ann.addNeuron(ActivationFunction.LINEAR));

        val outputNeurons = List.range(input, input + output)
            .map(i -> ann.addNeuron(ActivationFunction.SIGMOID));

        inputNeurons.forEach(graph::addVertex);
        outputNeurons.forEach(graph::addVertex);

        inputNeurons.forEach(inputNeuron -> {
            outputNeurons.forEach(outputNeuron -> {
                ann.addAxon(inputNeuron, outputNeuron, Math.random() * 2 - 1);
            });
        });

        return ann;
    }

    /**
     * Create a neural network from a graph
     * @param graph The graph to create the neural network from
     * @return The neural network
     */
    public static AdvancedNeuralNetwork from(
        DirectedAcyclicGraph<Neuron, Axon> graph
    ) {
        return new AdvancedNeuralNetwork(graph);
    }

    /**
     * Create an empty neural network
     * @return The empty neural network
     */
    public static AdvancedNeuralNetwork empty() {
        return new AdvancedNeuralNetwork(
            new DirectedAcyclicGraph<>(
                null,
                () -> new Axon(-1), // should never be called
                true
            )
        );
    }

    /**
     * Create a deep copy of the neural network
     *
     * @return The deep copy of the neural network
     */
    public AdvancedNeuralNetwork copy() {
        val newInstance = AdvancedNeuralNetwork.empty();

        for (val neuron : vertexSet()) {
            newInstance.addVertex(neuron);
        }

        for (val axon : edgeSet()) {
            val source = getEdgeSource(axon);
            val target = getEdgeTarget(axon);
            val weight = getEdgeWeight(axon);
            newInstance.addEdge(source, target, axon);
            newInstance.setEdgeWeight(axon, weight);
        }

        newInstance.axonId = axonId;
        newInstance.neuronId = neuronId;

        return newInstance;
    }
}
