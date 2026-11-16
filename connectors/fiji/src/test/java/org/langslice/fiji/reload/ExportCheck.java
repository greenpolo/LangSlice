package org.langslice.fiji.reload;

import ch.epfl.biop.atlas.aligner.*;
import ch.epfl.biop.registration.Registration;
import ch.epfl.biop.registration.source.bigwarp.BigWarpSource2DRegistration;
import com.google.gson.*;
import net.imglib2.realtransform.RealTransform;
import bdv.util.BoundedRealTransform;

import java.nio.file.*;
import java.util.*;

/**
 * Opens an ABBA project file LangSlice exported (tools/abba_export.py), in a JVM WITHOUT the LangSlice connector,
 * and writes where ABBA itself places given pixels of each slice in the atlas (ABBA's own pixel -> CCF transform,
 * the one its QuPath export writes), the slicing angles and the registrations each slice holds, so the export can
 * be compared with LangSlice's own maps. Opt-in, like {@link ReloadCheck}:
 * <pre>java -cp ... org.langslice.fiji.reload.ExportCheck CACHED_ATLAS PROJECT.abba POINTS.json OUT.json</pre>
 * POINTS.json: {"sections": {"file name": [[x, y], ...]}} in file pixels (centres at integers).
 */
public final class ExportCheck {
    static { net.imagej.patcher.LegacyInjector.preinit(); }

    public static void main(String[] args) throws Exception {
        Path cache = Paths.get(args[0]), project = Paths.get(args[1]), points = Paths.get(args[2]), out = Paths.get(args[3]);
        JsonObject wanted = JsonParser.parseString(Files.readString(points)).getAsJsonObject().getAsJsonObject("sections");
        net.imagej.ImageJ imagej = new net.imagej.ImageJ();
        ch.epfl.biop.atlas.AtlasLocationHelper.defaultCacheDir = cache.toFile();
        Object atlas = imagej.command().run(ch.epfl.biop.atlas.mouse.allen.ccfv3p1.command.AllenBrainAdultMouseAtlasCCF2017v3p1Command.class, true)
                .get().getOutput("ba");
        MultiSlicePositioner mp = (MultiSlicePositioner) imagej.command().run(ch.epfl.biop.atlas.aligner.command.ABBAStartCommand.class, true,
                "x_axis", "RL (Right-Left)", "y_axis", "SI (Superior-Inferior)", "z_axis", "AP (Anterior-Posterior)", "ba", atlas).get().getOutput("mp");
        if (!mp.loadState(project.toFile())) throw new AssertionError("ABBA did not load " + project);
        mp.waitForTasks();
        JsonObject result = new JsonObject();
        result.addProperty("rotate_x", mp.getReslicedAtlas().getRotateX());
        result.addProperty("rotate_y", mp.getReslicedAtlas().getRotateY());
        result.addProperty("slices", mp.getSlices().size());
        JsonObject sections = new JsonObject();
        for (SliceSources slice : mp.getSlices()) {
            slice.waitForEndOfTasks();
            String name = slice.getOriginalSources()[0].getSpimSource().getName();
            String file = name.replaceAll("-ch_\\d+$", "");
            JsonObject row = new JsonObject();
            row.addProperty("slice_name", slice.getName());
            row.addProperty("location", slice.getSlicingAxisPosition());
            row.addProperty("atlas_z", mp.toAtlasZ(slice.getSlicingAxisPosition()));
            JsonArray names = new JsonArray();
            slice.getRegistrationNames().forEach(names::add);
            row.add("registrations", names);
            boolean editable = true;
            for (CancelableAction action : new ArrayList<>(mp.getActionsFromSlice(slice))) {
                if (action instanceof RegisterSliceAction && action.isValid()) {
                    Registration<?> registration = ((RegisterSliceAction) action).getRegistration();
                    if (registration instanceof BigWarpSource2DRegistration) {
                        RealTransform warp = ((BigWarpSource2DRegistration) registration).getRealTransform();
                        if (warp instanceof BoundedRealTransform) warp = ((BoundedRealTransform) warp).getTransform();
                        editable &= bdv.util.RealTransformHelper.BigWarpFileFromRealTransform(warp) != null;
                    }
                }
            }
            row.addProperty("warp_editable", editable);
            JsonArray mapped = new JsonArray(), inverse = new JsonArray();
            if (wanted.has(file)) {
                RealTransform pixToCcf = slice.getSlicePixToCCFRealTransform();
                double[] target = new double[3];
                for (JsonElement point : wanted.getAsJsonArray(file)) {
                    JsonArray xy = point.getAsJsonArray();
                    pixToCcf.apply(new double[]{xy.get(0).getAsDouble(), xy.get(1).getAsDouble(), 0}, target);
                    JsonArray one = new JsonArray();
                    for (double v : target) one.add(v);
                    mapped.add(one);
                    if (pixToCcf instanceof net.imglib2.realtransform.InvertibleRealTransform) {
                        double[] back = new double[3];
                        ((net.imglib2.realtransform.InvertibleRealTransform) pixToCcf).applyInverse(back, new double[]{xy.get(0).getAsDouble(), xy.get(1).getAsDouble(), 0});
                        JsonArray two = new JsonArray();
                        for (double v : back) two.add(v);
                        inverse.add(two);
                    }
                }
            }
            row.add("ccf_mm", mapped);
            row.add("inverse_of_points", inverse);
            row.addProperty("transform_class", slice.getSlicePixToCCFRealTransform().getClass().getName());
            sections.add(file, row);
        }
        result.add("sections", sections);
        Files.writeString(out, new GsonBuilder().setPrettyPrinting().create().toJson(result));
        imagej.context().dispose();
        System.out.println("EXPORT_CHECK wrote " + out);
        System.exit(0);
    }
}
