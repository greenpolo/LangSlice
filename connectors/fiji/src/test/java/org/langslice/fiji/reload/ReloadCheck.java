package org.langslice.fiji.reload;

import bdv.util.BoundedRealTransform;
import ch.epfl.biop.atlas.aligner.*;
import ch.epfl.biop.registration.Registration;
import ch.epfl.biop.registration.source.bigwarp.BigWarpSource2DRegistration;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import net.imglib2.realtransform.RealTransform;

import java.nio.file.*;
import java.util.*;

/**
 * Reopens the project HostSessionSmokeTest saved, in a JVM WITHOUT the LangSlice connector: plain ABBA must
 * load LangSlice's affine and warp steps (ABBA's own AffineRegistration and BigWarp types) and BigWarp must
 * be able to edit the warp. Uses no connector class on purpose.
 */
public final class ReloadCheck {
    // ImageJ 1.x classes must be patched before any is loaded (ij1-patcher); Fiji's launcher does this itself.
    static { net.imagej.patcher.LegacyInjector.preinit(); }

    static void require(boolean condition, String message) { if (!condition) throw new AssertionError(message); }

    public static void main(String[] args) throws Exception {
        Path cache = Paths.get(args[0]), output = Paths.get(args[1]);
        boolean connector;
        try { Class.forName("org.langslice.fiji.AbbaHostSession"); connector = true; } catch (ClassNotFoundException absent) { connector = false; }
        require(!connector, "The connector must not be on this JVM's classpath");
        JsonObject expected = JsonParser.parseString(Files.readString(output.resolve("expected.json"))).getAsJsonObject();
        net.imagej.ImageJ imagej = new net.imagej.ImageJ();
        require(imagej.plugin().getPluginsOfClass("org.langslice.fiji.LangSliceService").isEmpty(), "No LangSlice plugin discovered");
        ch.epfl.biop.atlas.AtlasLocationHelper.defaultCacheDir = cache.toFile();
        Object atlas = imagej.command().run(ch.epfl.biop.atlas.mouse.allen.ccfv3p1.command.AllenBrainAdultMouseAtlasCCF2017v3p1Command.class, true)
                .get().getOutput("ba");
        MultiSlicePositioner mp = (MultiSlicePositioner) imagej.command().run(ch.epfl.biop.atlas.aligner.command.ABBAStartCommand.class, true,
                "x_axis", "RL (Right-Left)", "y_axis", "SI (Superior-Inferior)", "z_axis", "AP (Anterior-Posterior)", "ba", atlas).get().getOutput("mp");
        require(mp.loadState(output.resolve("smoke.abba").toFile()), "Saved project loads without LangSlice");
        mp.waitForTasks();
        require(mp.getSlices().size() == expected.get("slices").getAsInt(), "Both slices reopen");
        List<SliceSources> slices = new ArrayList<>(mp.getSlices());
        slices.sort(Comparator.comparingDouble(SliceSources::getSlicingAxisPosition));
        SliceSources one = slices.get(0);
        List<Registration<?>> active = new ArrayList<>();
        for (CancelableAction action : new ArrayList<>(mp.getActionsFromSlice(one))) {
            if (action instanceof RegisterSliceAction && action.isValid()) active.add(((RegisterSliceAction) action).getRegistration());
            else if (action instanceof DeleteLastRegistrationAction && action.isValid() && !active.isEmpty()) active.remove(active.size() - 1);
        }
        List<String> names = new ArrayList<>();
        for (Registration<?> registration : active) names.add(registration.getRegistrationName());
        require(names.equals(Arrays.asList("LangSlice affine", "LangSlice warp")) && one.getNumberOfRegistrations() == 2,
                "Affine and warp steps reopen in order: " + names + " / " + one.getRegistrationNames());
        require(one.getRegistrationNames().equals(Arrays.asList("AffineRegistration", "BigWarpSource2DRegistration")), "ABBA's own types: " + one.getRegistrationNames());
        RealTransform warp = ((BigWarpSource2DRegistration) active.get(1)).getRealTransform();
        if (warp instanceof BoundedRealTransform) warp = ((BoundedRealTransform) warp).getTransform();
        double amplitude = expected.get("amplitude").getAsDouble();
        double[] out = new double[3];
        for (int i = 0; i < 25; i++) {
            double x = (i % 5 - 2) * .6, y = (i / 5 - 2) * .5;
            warp.apply(new double[]{x + amplitude * Math.sin(y + .2), y + amplitude * .7 * Math.cos(x + .3), 0}, out);
            require(Math.hypot(out[0] - x, out[1] - y) < 1e-6, "Reloaded warp landmark " + i + ": " + Arrays.toString(out));
        }
        require(bdv.util.RealTransformHelper.BigWarpFileFromRealTransform(warp) != null, "BigWarp can edit the reloaded warp");
        require(Math.abs(mp.toAtlasZ(one.getSlicingAxisPosition()) - expected.get("position_mm").getAsDouble()) < 1e-9, "Position reopens");
        require(Math.abs(mp.getReslicedAtlas().getRotateX() - expected.get("rotate_x").getAsDouble()) < 1e-12
                && Math.abs(mp.getReslicedAtlas().getRotateY() - expected.get("rotate_y").getAsDouble()) < 1e-12, "Cutting angles reopen");
        imagej.context().dispose();
        System.out.println("HOST_RELOAD_PASS project reopened WITHOUT the connector: affine + BigWarp warp steps, warp landmarks, BigWarp-editable, position, angles");
        System.exit(0);
    }
}
