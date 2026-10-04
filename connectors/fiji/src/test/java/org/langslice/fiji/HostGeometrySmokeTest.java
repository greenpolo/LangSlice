package org.langslice.fiji;

import ch.epfl.biop.registration.source.affine.AffineRegistration;
import ch.epfl.biop.registration.source.bigwarp.BigWarpSource2DRegistration;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import net.imglib2.realtransform.AffineTransform3D;
import net.imglib2.realtransform.InvertibleRealTransform;
import net.imglib2.realtransform.InvertibleRealTransformSequence;
import net.imglib2.realtransform.RealTransform;
import org.scijava.Context;
import org.scijava.plugin.PluginService;

/** Real native transform/serialization checks: no atlas, model, or user project. */
public final class HostGeometrySmokeTest {
    private static void close(double[] expected, double[] actual, double tolerance, String message) {
        for (int i = 0; i < expected.length; i++) {
            if (!Double.isFinite(actual[i]) || Math.abs(expected[i] - actual[i]) > tolerance)
                throw new AssertionError(message + " axis " + i + ": " + actual[i] + " != " + expected[i]);
        }
    }
    private static double[] apply(RealTransform transform, double... point) {
        double[] result = new double[3]; transform.apply(point, result); return result;
    }
    private static JsonObject orientation(int rotation, boolean flip) {
        JsonObject value = new JsonObject(); value.addProperty("rotation_deg", rotation);
        value.addProperty("flip", flip); return value;
    }
    /** A warp row: [[x...],[y...]] or [[x, y], ...]; source = row's source_mm, target = row's target_mm. */
    static JsonObject warpRow(double[][] source, double[][] target, boolean rows) {
        JsonObject warp = new JsonObject();
        warp.add("source_mm", coordinates(source, rows)); warp.add("target_mm", coordinates(target, rows));
        warp.addProperty("record", "deformable/record-1"); warp.addProperty("max_error_mm", 0.004); warp.addProperty("points", source[0].length);
        return warp;
    }

    private static JsonArray coordinates(double[][] points, boolean rows) {
        JsonArray out = new JsonArray();
        if (rows) for (double[] axis : points) { JsonArray row = new JsonArray(); for (double v : axis) row.add(v); out.add(row); }
        else for (int i = 0; i < points[0].length; i++) { JsonArray p = new JsonArray(); p.add(points[0][i]); p.add(points[1][i]); out.add(p); }
        return out;
    }

    /**
     * The warp step: same pull-back as the legacy spline rows (target point -> source point), in both coordinate
     * layouts; a BigWarp step that survives ABBA's serialization and that BigWarp can reopen for editing.
     */
    static int warpChecks(double[][] source, double[][] target) throws Exception {
        int probes = 0;
        InvertibleRealTransform byRows = AbbaHostSession.warpTransform(warpRow(source, target, true));
        InvertibleRealTransform byPoints = AbbaHostSession.warpTransform(warpRow(source, target, false));
        for (int i = 0; i < source[0].length; i++) {
            double[] expected = {source[0][i], source[1][i], 0};
            close(expected, apply(byRows, target[0][i], target[1][i], 0), 1e-8, "Warp maps target_mm to source_mm (rows)");
            close(expected, apply(byPoints, target[0][i], target[1][i], 0), 1e-8, "Warp maps target_mm to source_mm (points)");
            probes += 2;
        }
        // Exactly the legacy pull-back with the orientation left out (the affine step carries it).
        InvertibleRealTransform legacy = AbbaHostSession.tps(target, source);
        for (int ix = -4; ix <= 4; ix++) for (int iy = -4; iy <= 4; iy++) {
            double x = ix * .31, y = iy * .27;
            close(apply(legacy, x, y, 0), apply(byRows, x, y, 0), 1e-12, "Warp equals the legacy TPS off landmarks");
            probes++;
        }
        try { AbbaHostSession.warpTransform(JsonParser.parseString("{\"source_mm\":[[0,1],[0,1]],\"target_mm\":[[0,1],[0,1]]}").getAsJsonObject());
              throw new AssertionError("Two points cannot make a warp"); } catch (IllegalArgumentException expected) { }
        JsonObject miscounted = warpRow(source, target, true); miscounted.addProperty("points", 3);
        try { AbbaHostSession.warpTransform(miscounted); throw new AssertionError("Point count is checked"); } catch (IllegalArgumentException expected) { }
        try (Context context = new Context(PluginService.class, sc.fiji.persist.IObjectScijavaAdapterService.class)) {
            BigWarpSource2DRegistration registration = new BigWarpSource2DRegistration();
            registration.setScijavaContext(context); registration.setRealTransform(byRows);
            String serialized = registration.getTransform();
            BigWarpSource2DRegistration reopened = new BigWarpSource2DRegistration();
            reopened.setScijavaContext(context); reopened.setTransform(serialized);
            if (!reopened.isRegistrationDone()) throw new AssertionError("Reloaded warp incomplete");
            for (int ix = -5; ix <= 5; ix++) for (int iy = -4; iy <= 4; iy++) {
                double x = ix * .23, y = iy * .29;
                close(apply(byRows, x, y, 0), apply(reopened.getRealTransform(), x, y, 0), 1e-12, "Warp serialization off-landmark equivalence");
                probes++;
            }
            String landmarks = bdv.util.RealTransformHelper.BigWarpFileFromRealTransform(reopened.getRealTransform());
            if (landmarks == null) throw new AssertionError("BigWarp cannot reopen the warp for editing");
            java.util.List<String> lines = java.nio.file.Files.readAllLines(java.nio.file.Paths.get(landmarks));
            if (lines.size() != source[0].length) throw new AssertionError("BigWarp landmark file has " + lines.size() + " rows");
        }
        return probes;
    }

    /** ABBA rotations from host_angles and back: rotateX = -radians(pitch), rotateY = -radians(yaw). */
    static int angleChecks() {
        JsonObject angles = new JsonObject(); angles.addProperty("pitch_deg", 10.0); angles.addProperty("yaw_deg", -5.0);
        double[] rotation = AbbaHostSession.rotations(angles);
        close(new double[]{-Math.toRadians(10), Math.toRadians(5)}, rotation, 1e-15, "host_angles signs");
        JsonObject back = AbbaHostSession.anglesDeg(rotation[0], rotation[1]);
        close(new double[]{10.0, -5.0}, new double[]{back.get("pitch_deg").getAsDouble(), back.get("yaw_deg").getAsDouble()}, 1e-12, "angles_deg round trip");
        JsonObject flat = AbbaHostSession.anglesDeg(0, 0);
        if (flat.get("pitch_deg").getAsDouble() != 0 || flat.get("yaw_deg").getAsDouble() != 0) throw new AssertionError("Flat session sends zero angles");
        return 3;
    }

    /** A row kept after a failure, merged with the next checkpoint's row for the same section. */
    static void mergeChecks() {
        JsonObject kept = JsonParser.parseString("{\"id\":\"a\",\"position_mm\":1,\"spline_source_mm\":[],\"spline_target_mm\":[],\"warp\":null}").getAsJsonObject();
        JsonObject next = JsonParser.parseString("{\"id\":\"a\",\"affine_mm\":[[1,0,0,0],[0,1,0,0],[0,0,1,0]],\"warp\":{\"source_mm\":[]}}").getAsJsonObject();
        JsonObject merged = AbbaHostSession.merge(kept, next);
        if (merged.has("spline_source_mm") || merged.has("spline_target_mm") || !merged.has("affine_mm")
                || merged.get("position_mm").getAsDouble() != 1 || !merged.get("warp").isJsonObject())
            throw new AssertionError("Merge: newer keys win and a new affine replaces a kept spline: " + merged);
        if (kept.has("affine_mm")) throw new AssertionError("Merge leaves its inputs alone");
    }

    public static void main(String[] args) throws Exception {
        if (args.length == 2 && args[0].equals("--render-setup")) {
            javax.swing.SwingUtilities.invokeAndWait(() -> {
                try {
                    java.lang.reflect.Constructor<SetupDialog> constructor = SetupDialog.class.getDeclaredConstructor(Runnable.class);
                    constructor.setAccessible(true);
                    SetupDialog dialog = constructor.newInstance((Runnable) null);
                    java.awt.Container panel = dialog.getContentPane();
                    java.awt.image.BufferedImage image = new java.awt.image.BufferedImage(
                            panel.getWidth(), panel.getHeight(), java.awt.image.BufferedImage.TYPE_INT_RGB);
                    java.awt.Graphics2D graphics = image.createGraphics();
                    panel.printAll(graphics); graphics.dispose();
                    javax.imageio.ImageIO.write(image, "png", new java.io.File(args[1]));
                    dialog.dispose();
                } catch (Exception failure) { throw new RuntimeException(failure); }
            });
            return;
        }
        // Includes unequal scales, shear and translation: order mistakes cannot
        // hide behind isotropic or identity test transforms.
        AffineTransform3D affine = new AffineTransform3D();
        affine.set(1.2, .17, 0, .42, -.08, .83, 0, -.31, 0, 0, 1, 0);
        int probes = 0;
        for (int rotation : new int[]{0,90,180,270}) for (boolean flip : new boolean[]{false,true}) {
            AffineTransform3D orient = AbbaHostSession.orientation(orientation(rotation, flip));
            AffineTransform3D complete = affine.copy(); complete.concatenate(orient);
            AffineTransform3D restored = AffineRegistration.stringToAffineTransform3D(
                    AffineRegistration.affineTransform3DToString(complete));
            double radians = Math.toRadians(-rotation);
            for (int ix = -3; ix <= 3; ix++) for (int iy = -2; iy <= 2; iy++) {
                double x = ix * .47, y = iy * .39;
                double ox = Math.cos(radians)*x - Math.sin(radians)*y;
                double oy = Math.sin(radians)*x + Math.cos(radians)*y;
                if (flip) ox = -ox;
                double[] expected = {1.2*ox + .17*oy + .42, -.08*ox + .83*oy - .31, 0};
                close(expected, apply(complete,x,y,0), 1e-12, "Affine orientation composition");
                close(expected, apply(restored,x,y,0), 1e-12, "Affine native string reload");
                probes++;
            }
        }
        double[][] source = new double[2][25], target = new double[2][25];
        for (int i=0; i<25; i++) {
            double x=(i%5-2)*.6, y=(i/5-2)*.6;
            source[0][i]=x; source[1][i]=y;
            target[0][i]=x+.12*Math.sin(y+.2); target[1][i]=y+.08*Math.cos(x+.3);
        }
        InvertibleRealTransform forward = AbbaHostSession.tps(source,target);
        for (int i=0; i<25; i++)
            close(new double[]{target[0][i],target[1][i],0},
                    apply(forward,source[0][i],source[1][i],0),1e-8,"TPS landmark direction");
        try (Context context = new Context(PluginService.class, sc.fiji.persist.IObjectScijavaAdapterService.class)) {
            for (int rotation : new int[]{0,90,180,270}) for (boolean flip : new boolean[]{false,true}) {
                AffineTransform3D orient = AbbaHostSession.orientation(orientation(rotation,flip));
                InvertibleRealTransformSequence pullback = new InvertibleRealTransformSequence();
                pullback.add(AbbaHostSession.tps(target,source)); pullback.add(orient.inverse());
                BigWarpSource2DRegistration registration = new BigWarpSource2DRegistration();
                registration.setScijavaContext(context); registration.setRealTransform(pullback);
                String serialized = registration.getTransform();
                BigWarpSource2DRegistration reopened = new BigWarpSource2DRegistration();
                reopened.setScijavaContext(context); reopened.setTransform(serialized);
                if (!reopened.isRegistrationDone()) throw new AssertionError("Reloaded registration incomplete");
                for (int i=0; i<25; i++) {
                    double[] expected = apply(orient.inverse(),source[0][i],source[1][i],0);
                    close(expected,apply(reopened.getRealTransform(),target[0][i],target[1][i],0),
                            1e-8,"Reloaded TPS plus inverse orientation");
                    probes++;
                }
                for (int ix=-5; ix<=5; ix++) for (int iy=-4; iy<=4; iy++) {
                    double x=ix*.23,y=iy*.29;
                    close(apply(pullback,x,y,0),apply(reopened.getRealTransform(),x,y,0),
                            1e-12,"TPS serialization off-landmark equivalence");
                    probes++;
                }
            }
        }
        probes += warpChecks(source, target);
        probes += angleChecks();
        mergeChecks();
        System.out.println("Native host geometry passed: " + probes + " affine/TPS/warp/angle/serialization probes.");
    }
}
