package org.langslice.fiji;

import ch.epfl.biop.atlas.aligner.CancelableAction;
import ch.epfl.biop.atlas.aligner.MultiSlicePositioner;
import ch.epfl.biop.atlas.aligner.ReslicedAtlas;
import ch.epfl.biop.atlas.aligner.SliceSources;

import java.awt.Graphics2D;

/**
 * The stack-wide atlas cutting angles as an undoable ABBA action. ABBA's own "Set Atlas Slicing Angles"
 * is not undoable; this action sets the same {@link ReslicedAtlas} rotations, and its cancel restores the
 * previous ones, so a checkpoint that changes the angles is still ONE ABBA undo step. It belongs to no slice,
 * so ABBA never writes it into a saved project (the angles themselves are saved as part of the state).
 */
final class SlicingAnglesAction extends CancelableAction {
    private final double rotateX, rotateY;
    private double previousX, previousY;

    SlicingAnglesAction(MultiSlicePositioner mp, double rotateX, double rotateY) {
        super(mp);
        this.rotateX = rotateX;
        this.rotateY = rotateY;
        hide();
    }

    @Override public SliceSources getSliceSources() { return null; }

    @Override protected boolean run() {
        ReslicedAtlas atlas = getMP().getReslicedAtlas();
        previousX = atlas.getRotateX();
        previousY = atlas.getRotateY();
        atlas.setRotateX(rotateX);
        atlas.setRotateY(rotateY);
        getMP().stateHasBeenChanged();
        return atlas.getRotateX() == rotateX && atlas.getRotateY() == rotateY;
    }

    @Override protected boolean cancel() {
        ReslicedAtlas atlas = getMP().getReslicedAtlas();
        atlas.setRotateX(previousX);
        atlas.setRotateY(previousY);
        getMP().stateHasBeenChanged();
        return true;
    }

    @Override public void drawAction(Graphics2D g, double px, double py, double scale) { }

    @Override public String toString() { return "LangSlice cutting angles"; }
}
