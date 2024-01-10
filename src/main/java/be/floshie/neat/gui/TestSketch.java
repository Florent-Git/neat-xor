package be.floshie.neat.gui;

import be.floshie.neat.ai.graph.Axon;
import be.floshie.neat.ai.graph.Neuron;
import lombok.RequiredArgsConstructor;
import lombok.val;
import org.jgrapht.Graph;
import org.jgrapht.alg.drawing.FRLayoutAlgorithm2D;
import org.jgrapht.alg.drawing.LayoutAlgorithm2D;
import org.jgrapht.alg.drawing.model.Box2D;
import org.jgrapht.alg.drawing.model.LayoutModel2D;
import org.jgrapht.alg.drawing.model.MapLayoutModel2D;
import processing.core.PApplet;

import java.util.concurrent.atomic.AtomicReference;

@RequiredArgsConstructor
public class TestSketch extends PApplet {
    private final Graph<Neuron, Axon> directedGraph;
    private LayoutAlgorithm2D<Neuron, Axon> layoutAlgorithm;
    private LayoutModel2D<Neuron> layoutModel;

    private final static int circleRadius = 50;

    @Override
    public void settings() {
        size(640, 360);
    }

    @Override
    public void setup() {
        layoutAlgorithm = new FRLayoutAlgorithm2D<>(4);
        layoutModel = new MapLayoutModel2D<>(Box2D.of(0, 0, width, height));
        layoutAlgorithm.layout(directedGraph, layoutModel);
    }

    @Override
    public void draw() {
        background(0);
        stroke(255);
        strokeWeight(2);
        noFill();

        for (val vertex : directedGraph.vertexSet()) {
            val position = layoutModel.get(vertex);
            ellipse((float) position.getX(), (float) position.getY(), 20, 20);
            // Write the neuron's id in the center of the circle
            fill(255);
            textAlign(CENTER, CENTER);
            text(vertex.getId(), (float) position.getX(), (float) position.getY());
            noFill();
        }

        for (val edge : directedGraph.edgeSet()) {
            // Base the color on the weight of the axon
            // It is a value between -1 and 1, so we need to map it to a gradient between red and blue
            val weight = directedGraph.getEdgeWeight(edge);
            val color = map((float) weight, -1, 1, 0, 255);
            stroke(color, 0, 255 - color);

            val sourcePosition = layoutModel.get(directedGraph.getEdgeSource(edge));
            val targetPosition = layoutModel.get(directedGraph.getEdgeTarget(edge));
            // Draw an arrow from the source to the target with wings and taking the circle radius into account
            drawArrow(
                (float) sourcePosition.getX(),
                (float) sourcePosition.getY(),
                (float) targetPosition.getX(),
                (float) targetPosition.getY()
            );
        }
    }

    private void drawArrow(float x1, float y1, float x2, float y2) {
        // Calculate the angle of the line
        float angle = atan2(y1 - y2, x1 - x2);
        // Calculate the position of the arrow
        float x3 = x2 + cos(angle) * (float) circleRadius;
        float y3 = y2 + sin(angle) * (float) circleRadius;
        // Draw the line
        line(x1, y1, x3, y3);
        // Draw the arrow wings
        line(x3, y3, x3 + cos(angle - PI / 4) * 10, y3 + sin(angle - PI / 4) * 10);
        line(x3, y3, x3 + cos(angle + PI / 4) * 10, y3 + sin(angle + PI / 4) * 10);
    }
}
