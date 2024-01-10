package be.floshie.neat.ai;

import be.floshie.neat.ai.graph.Axon;
import be.floshie.neat.ai.graph.Neuron;
import io.vavr.collection.HashMap;
import io.vavr.collection.List;
import io.vavr.collection.Map;
import lombok.val;
import org.jgrapht.Graph;
import org.jgrapht.graph.DirectedAcyclicGraph;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.stream.Collectors;
import java.util.stream.Stream;

import static be.floshie.neat.ai.ActivationFunction.LINEAR;
import static org.junit.jupiter.api.Assertions.*;

class AdvancedNeuralNetworkTest {
    private static AdvancedNeuralNetwork ann1;

    @BeforeEach
    void setUp() {
        ann1 = AdvancedNeuralNetwork.empty();

        val n1 = ann1.addNeuron(LINEAR);
        val n2 = ann1.addNeuron(LINEAR);
        val n3 = ann1.addNeuron(LINEAR);
        val n4 = ann1.addNeuron(LINEAR);
        val n5 = ann1.addNeuron(LINEAR);

        ann1.addAxon(n1, n3, 0.5);
        ann1.addAxon(n1, n4, 0.5);
        ann1.addAxon(n2, n3, 0.5);
        ann1.addAxon(n2, n4, 0.5);
        ann1.addAxon(n3, n5, 0.5);
        ann1.addAxon(n4, n3, 0.5);
    }


    // Create multiple test cases
    static Stream<Arguments> givenGraph01_whenGivingInput_thenOutputIsCorrect() {
        return Stream.of(
                Arguments.of(1, 1, .75),
                Arguments.of(2, 3, 1.875),
                Arguments.of(4, -2, .75),
                Arguments.of(.75, -.33, .1575),
                Arguments.of(2, 2, 1.5)
        );
    }

    @ParameterizedTest
    @MethodSource
    void givenGraph01_whenGivingInput_thenOutputIsCorrect(
            double input1, double input2, double output5
    ) {
        Map<Integer, Double> input = HashMap.of(
            0, input1,
            1, input2
        );

        List<Double> output = ann1.feedForward(input);

        assertAll(
            () -> assertEquals(output5, output.get(0))
        );
    }

    @Test
    void givenGraph01_whenCopied_thenReturnedValueIsNotTheSameInstance() {
        val ann1Copy = ann1.copy();

        assertNotSame(ann1, ann1Copy);
        assertGraphEquals(ann1, ann1Copy);
    }

    private <V, E> void assertGraphEquals(Graph<V, E> expected, Graph<V, E> actual) {
        assertEquals(expected.vertexSet(), actual.vertexSet());
        assertEquals(expected.edgeSet(), actual.edgeSet());
        assertEquals(expected.edgeSet().stream().map(expected::getEdgeWeight).collect(Collectors.toList()),
            actual.edgeSet().stream().map(actual::getEdgeWeight).collect(Collectors.toList()));
    }
}