package be.floshie.neat.ai.graph;

import lombok.*;

@Getter
@ToString
@EqualsAndHashCode
public class Axon {
    private final int id;
    private final int innovation;

    @With
    @EqualsAndHashCode.Exclude
    private final boolean enabled;

    @Setter
    @EqualsAndHashCode.Exclude
    private double weight;

    public Axon(int id) {
        this(id, true);
    }

    public Axon(int id, boolean enabled) {
        this(id, InnovationNumber.innovationCounter++, enabled);
    }

    protected Axon(int id, int innovation, boolean enabled) {
        this(id, innovation, enabled, 1.0);
    }

    protected Axon(int id, int innovation, boolean enabled, double weight) {
        this.id = id;
        this.enabled = enabled;
        this.innovation = innovation;
        this.weight = weight;
    }
}
