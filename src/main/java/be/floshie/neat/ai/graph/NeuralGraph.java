package be.floshie.neat.ai.graph;

import org.jgrapht.Graph;

/**
 * Sort of a type alias for the graph without the need to
 * write the generic types every time.
 * Especially useful to use with Lombok's @{@link lombok.experimental.Delegate} annotation.
 */
public interface NeuralGraph extends Graph<Neuron, Axon> {
}