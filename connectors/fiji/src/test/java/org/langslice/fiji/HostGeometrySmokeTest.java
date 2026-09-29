package org.langslice.fiji;

import ch.epfl.biop.registration.sourceandconverter.affine.AffineRegistration;
import ch.epfl.biop.registration.sourceandconverter.bigwarp.SacBigWarp2DRegistration;
import com.google.gson.JsonObject;
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
                SacBigWarp2DRegistration registration = new SacBigWarp2DRegistration();
                registration.setScijavaContext(context); registration.setRealTransform(pullback);
                String serialized = registration.getTransform();
                SacBigWarp2DRegistration reopened = new SacBigWarp2DRegistration();
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
        System.out.println("Native host geometry passed: " + probes + " affine/TPS/serialization probes.");
    }
}
