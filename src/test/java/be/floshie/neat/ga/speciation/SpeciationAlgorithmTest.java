package be.floshie.neat.ga.speciation;

import be.floshie.neat.ai.ActivationFunction;
import be.floshie.neat.ai.AdvancedNeuralNetwork;
import be.floshie.neat.ai.graph.Axon;
import be.floshie.neat.ga.Individual;
import lombok.val;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class SpeciationAlgorithmTest {
    private static class TestAxon extends Axon {
        private static int idCounter = 0;

        public TestAxon(int innovation, boolean enabled) {
            super(idCounter++, innovation, enabled);
        }
    }
    
    private Individual createParent1() {
        val parent1 = AdvancedNeuralNetwork.empty();
        
        val n1 = parent1.addNeuron(ActivationFunction.LINEAR);
        val n2 = parent1.addNeuron(ActivationFunction.LINEAR);
        val n3 = parent1.addNeuron(ActivationFunction.LINEAR);
        val n4 = parent1.addNeuron(ActivationFunction.LINEAR);
        val n5 = parent1.addNeuron(ActivationFunction.LINEAR);
        
        parent1.addEdge(n1, n4, new TestAxon(1, true));
        parent1.addEdge(n2, n4, new TestAxon(2, true));
        parent1.addEdge(n3, n4, new TestAxon(3, true));
        parent1.addEdge(n2, n5, new TestAxon(4, true));
        parent1.addEdge(n5, n4, new TestAxon(5, true));
        parent1.addEdge(n1, n5, new TestAxon(8, true));
        
        return new Individual(parent1);
    }

    private Individual createParent2() {
        val parent2 = AdvancedNeuralNetwork.empty();

        val n1 = parent2.addNeuron(ActivationFunction.LINEAR);
        val n2 = parent2.addNeuron(ActivationFunction.LINEAR);
        val n3 = parent2.addNeuron(ActivationFunction.LINEAR);
        val n4 = parent2.addNeuron(ActivationFunction.LINEAR);
        val n5 = parent2.addNeuron(ActivationFunction.LINEAR);
        val n6 = parent2.addNeuron(ActivationFunction.LINEAR);

        parent2.addEdge(n1, n4, new TestAxon(1, true));
        parent2.addEdge(n2, n4, new TestAxon(2, true));
        parent2.addEdge(n3, n4, new TestAxon(3, true));
        parent2.addEdge(n2, n5, new TestAxon(4, true));
        parent2.addEdge(n5, n4, new TestAxon(5, true));
        parent2.addEdge(n5, n6, new TestAxon(6, true));
        parent2.addEdge(n6, n4, new TestAxon(7, true));
        parent2.addEdge(n3, n5, new TestAxon(9, true));
        parent2.addEdge(n1, n6, new TestAxon(10, true));

        return new Individual(parent2);
    }
    
    @Test
    void getDistance() {
        val parent1 = createParent1();
        val parent2 = createParent2();

        val speciationAlgorithm = new SpeciationAlgorithm(0.1, 0.2, 0.3, 1);
        assertEquals(0.08, speciationAlgorithm.getDistance(parent1, parent2), 0.01);
    }
}