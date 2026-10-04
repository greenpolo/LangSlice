package org.langslice.fiji;

import bdv.viewer.SourceAndConverter;
import ch.epfl.biop.atlas.aligner.*;
import ch.epfl.biop.atlas.aligner.action.MarkActionSequenceBatchAction;
import ch.epfl.biop.registration.Registration;
import ch.epfl.biop.registration.plugin.SimpleRegistrationPlugin;
import ch.epfl.biop.registration.plugin.SimpleRegistrationWrapper;
import ch.epfl.biop.registration.sourceandconverter.affine.AffineRegistration;
import ch.epfl.biop.registration.sourceandconverter.bigwarp.SacBigWarp2DRegistration;
import ch.epfl.biop.sourceandconverter.processor.SourcesChannelsSelect;
import ch.epfl.biop.sourceandconverter.processor.SourcesProcessorHelper;
import com.google.gson.*;
import ij.ImagePlus;
import ij.ImageStack;
import ij.io.FileSaver;
import ij.measure.Calibration;
import ij.process.ImageProcessor;
import net.imglib2.realtransform.*;
import net.imglib2.realtransform.inverse.WrappedIterativeInvertibleRealTransform;
import org.scijava.plugin.PluginService;

import java.io.IOException;
import java.nio.file.*;
import java.util.*;

/** Calibrated snapshots and native, undoable registrations in an existing ABBA session. */
final class AbbaHostSession {
    final MultiSlicePositioner mp;
    final Path folder;
    final LinkedHashMap<String, SliceSources> slices = new LinkedHashMap<>();
    private final Map<String, Integer> baseline = new HashMap<>();
    private final Map<String, Registration<SourceAndConverter<?>[]>> owned = new HashMap<>();
    private final Map<String, JsonObject> transforms = new HashMap<>();
    private double axisScale, axisOffset;

    AbbaHostSession(MultiSlicePositioner mp, Path folder) {
        this.mp = mp;
        this.folder = folder;
    }

    /**
     * Exports the listed slices as calibrated snapshots and builds the linear.run request.
     * Each snapshot has one page per exported channel, in the order given. Nothing in ABBA changes.
     */
    JsonObject prepare(JsonObject spec, List<SliceSources> chosen, List<Integer> channels, double pixelSize,
            Map<SliceSources, String> damaged, boolean lockRegistered) throws IOException {
        validateSession();
        if (!Double.isFinite(pixelSize) || pixelSize <= 0 || channels.isEmpty() || channels.stream().anyMatch(c -> c < 0))
            throw new IllegalArgumentException("Choose a positive pixel size and at least one channel.");
        mp.waitForTasks();
        List<SliceSources> selected = new ArrayList<>(chosen);
        selected.sort(Comparator.comparingDouble(SliceSources::getSlicingAxisPosition));
        if (selected.isEmpty()) throw new IllegalArgumentException("Import sections before running LangSlice.");
        for (SliceSources slice : selected) {
            if (!mp.getSlices().contains(slice))
                throw new IllegalArgumentException("A listed slice was removed from ABBA. Reopen LangSlice Registration.");
            for (int channel : channels)
                if (channel >= slice.getRegisteredSources().length)
                    throw new IllegalArgumentException("Channel " + (channel + 1) + " is missing from " + slice.getName());
        }
        measureAxis(selected.get(0));
        double[] half = frame(pixelSize);
        Files.createDirectories(folder);
        JsonObject positions = new JsonObject();
        JsonArray registered = new JsonArray(), locked = new JsonArray();
        JsonObject marked = new JsonObject();
        JsonObject mapping = new JsonObject();
        for (int index = 0; index < selected.size(); index++) {
            checkInterrupted();
            SliceSources slice = selected.get(index);
            String name = String.format(Locale.ROOT, "section_%04d.tif", index + 1);
            ImagePlus image = snapshot(slice, channels, half[0], half[1], pixelSize / 1000.0);
            try { save(image, folder.resolve(name)); } finally { image.close(); }
            slices.put(name, slice);
            baseline.put(name, slice.getNumberOfRegistrations());
            if (slice.getNumberOfRegistrations() > 0) {
                registered.add(name);
                // Their snapshot already carries the registration: the agent keeps its in-plane geometry.
                if (lockRegistered) locked.add(name);
            }
            if (damaged.containsKey(slice)) marked.addProperty(name, damaged.get(slice) == null ? "" : damaged.get(slice));
            positions.addProperty(name, (slice.getSlicingAxisPosition() - axisOffset) / axisScale);
            mapping.addProperty(name, slice.getName());
        }
        Files.writeString(folder.resolve("abba_sections.json"), new GsonBuilder().setPrettyPrinting().create().toJson(mapping));
        JsonObject request = new JsonObject();
        request.addProperty("image_folder", folder.toString());
        request.addProperty("pixel_size_um", pixelSize);
        request.add("positions_mm", positions);
        request.add("registered_slices", registered);
        request.add("locked", locked);
        request.add("damaged", marked);
        request.add("spec", spec);
        return request;
    }

    /** Half extents of the snapshot frame: ABBA's image region, centred, on the pixel grid. */
    private double[] frame(double pixelSize) {
        double[] roi = mp.getROI();
        double spacing = pixelSize / 1000.0;
        double halfX = Math.ceil(Math.max(Math.abs(roi[0]), Math.abs(roi[0] + roi[2])) / spacing) * spacing;
        double halfY = Math.ceil(Math.max(Math.abs(roi[1]), Math.abs(roi[1] + roi[3])) / spacing) * spacing;
        if (!Double.isFinite(halfX + halfY) || halfX <= 0 || halfY <= 0)
            throw new IllegalArgumentException("ABBA's image region has no usable extent.");
        return new double[]{halfX, halfY};
    }

    /** One slice for the Preprocessing preview: the same framing as a run, written to file; returns its display. */
    java.awt.image.BufferedImage exportPreview(SliceSources slice, List<Integer> channels, double pixelSize, Path file) throws IOException {
        if (!Double.isFinite(pixelSize) || pixelSize <= 0) throw new IllegalArgumentException("Choose a positive pixel size.");
        double[] half = frame(pixelSize);
        ImagePlus image = snapshot(slice, channels, half[0], half[1], pixelSize / 1000.0);
        try { save(image, file); return display(image); } finally { image.close(); }
    }

    /** Registered channels in the given order, one page each, on ABBA's calibrated grid. */
    static ImagePlus snapshot(SliceSources slice, List<Integer> channels, double halfX, double halfY, double spacing) {
        if (channels.size() == 1)
            return SliceToImagePlus.export(slice, new SourcesChannelsSelect(channels.get(0)), -halfX, -halfY, 2 * halfX, 2 * halfY, spacing, 0, true);
        List<ImageProcessor> pages = new ArrayList<>();
        Calibration calibration = null;
        for (int channel : channels) {
            ImagePlus page = SliceToImagePlus.export(slice, new SourcesChannelsSelect(channel), -halfX, -halfY, 2 * halfX, 2 * halfY, spacing, 0, true);
            try {
                pages.add(page.getProcessor().duplicate());
                if (calibration == null) calibration = page.getCalibration().copy();
            } finally { page.close(); }
        }
        return stack(slice.getName(), pages, calibration);
    }

    /** Pages of different pixel types (or colour pages) become 32-bit grayscale so they share one TIFF. */
    static ImagePlus stack(String title, List<ImageProcessor> pages, Calibration calibration) {
        boolean convert = pages.stream().mapToInt(ImageProcessor::getBitDepth).distinct().count() > 1
                || pages.stream().anyMatch(page -> page.getBitDepth() == 24);
        ImageStack stack = new ImageStack(pages.get(0).getWidth(), pages.get(0).getHeight());
        for (int i = 0; i < pages.size(); i++) stack.addSlice("channel " + (i + 1), convert ? pages.get(i).convertToFloat() : pages.get(i));
        ImagePlus image = new ImagePlus(title, stack);
        image.setDimensions(pages.size(), 1, 1);
        if (calibration != null) image.setCalibration(calibration);
        return image;
    }

    static void save(ImagePlus image, Path file) throws IOException {
        FileSaver saver = new FileSaver(image);
        boolean saved = image.getStackSize() > 1 ? saver.saveAsTiffStack(file.toString()) : saver.saveAsTiff(file.toString());
        if (!saved) throw new IOException("Could not save section snapshot " + file.getFileName());
    }

    /** A plain display of the exported pages: gray for one channel, coloured overlay for several. */
    static java.awt.image.BufferedImage display(ImagePlus image) {
        int n = image.getStackSize(), w = image.getWidth(), h = image.getHeight();
        int[][] colors = n == 1 ? new int[][]{{255, 255, 255}} : new int[][]{{0, 255, 0}, {255, 0, 255}, {0, 160, 255}, {255, 160, 0}, {255, 255, 255}};
        float[] red = new float[w * h], green = new float[w * h], blue = new float[w * h];
        for (int page = 1; page <= n; page++) {
            float[] pixels = (float[]) image.getStack().getProcessor(page).convertToFloatProcessor().getPixels();
            float[] sorted = pixels.clone(); Arrays.sort(sorted);
            int finite = sorted.length; while (finite > 0 && Float.isNaN(sorted[finite - 1])) finite--;
            if (finite == 0) continue;
            float low = sorted[(int) (0.0035 * (finite - 1))], high = sorted[(int) (0.9965 * (finite - 1))];
            if (!(high > low)) high = low + 1;
            int[] color = colors[(page - 1) % colors.length];
            for (int i = 0; i < pixels.length; i++) {
                float value = Float.isNaN(pixels[i]) ? 0 : Math.max(0, Math.min(1, (pixels[i] - low) / (high - low)));
                red[i] += value * color[0]; green[i] += value * color[1]; blue[i] += value * color[2];
            }
        }
        java.awt.image.BufferedImage out = new java.awt.image.BufferedImage(w, h, java.awt.image.BufferedImage.TYPE_INT_RGB);
        for (int i = 0; i < red.length; i++)
            out.setRGB(i % w, i / w, (Math.min(255, (int) red[i]) << 16) | (Math.min(255, (int) green[i]) << 8) | Math.min(255, (int) blue[i]));
        return out;
    }

    private void validateSession() {
        String atlas = mp.getAtlas().getName();
        if (!atlas.equals("Adult Mouse Brain - Allen Brain Atlas V3p1") && !atlas.equals("Adult Mouse Brain - Allen Brain Atlas V3"))
            throw new IllegalArgumentException("LangSlice currently requires ABBA's Allen Mouse V3 atlas.");
        double x = mp.getReslicedAtlas().getRotateX(), y = mp.getReslicedAtlas().getRotateY();
        if (!Double.isFinite(x + y) || Math.abs(x) > 1e-8 || Math.abs(y) > 1e-8)
            throw new IllegalArgumentException("Set atlas cutting angles to zero before this run.");
    }

    /** Uses ABBA's fixed coordinate images, not an assumed anterior-edge offset. */
    private void measureAxis(SliceSources slice) {
        double ap1 = probeAt(slice,3.0), ap2 = probeAt(slice,9.0);
        axisScale = (3.0-9.0)/(ap1-ap2);
        axisOffset = 3.0-axisScale*ap1;
        if (!Double.isFinite(axisScale+axisOffset) || Math.abs(axisScale)<0.5 || Math.abs(axisScale)>2)
            throw new IllegalStateException("Could not calibrate ABBA's AP axis. Use a flat coronal session.");
    }

    private double probeAt(SliceSources slice, double z) {
        final double[] measured = {Double.NaN};
        SimpleRegistrationPlugin probe = new SimpleRegistrationPlugin() {
            public double getVoxelSizeInMicron() { return 40; }
            public void setRegistrationParameters(Map<String, String> parameters) { }
            public InvertibleRealTransform register(ImagePlus fixed, ImagePlus moving, ImagePlus fm, ImagePlus mm) {
                float center = fixed.getStack().getProcessor(1).getf(fixed.getWidth()/2, fixed.getHeight()/2);
                float left = fixed.getStack().getProcessor(1).getf(fixed.getWidth()/3, fixed.getHeight()/2);
                float above = fixed.getStack().getProcessor(1).getf(fixed.getWidth()/2, fixed.getHeight()/3);
                if (Float.isFinite(center) && Math.abs(center-left)<1e-4 && Math.abs(center-above)<1e-4)
                    measured[0] = center;
                return new AffineTransform3D();
            }
        };
        SimpleRegistrationWrapper wrapper = new SimpleRegistrationWrapper("LangSlice axis probe",probe);
        wrapper.setScijavaContext(mp.getContext());
        // This is the same -z translation as SourcesZOffset(slice), without moving the slice.
        AffineTransform3D offset = new AffineTransform3D();
        offset.translate(0,0,-z);
        double[] roi = mp.getROI();
        Map<String,Object> parameters = new HashMap<>();
        parameters.put("px",roi[0]); parameters.put("py",roi[1]);
        parameters.put("sx",roi[2]); parameters.put("sy",roi[3]); parameters.put("pz",0);
        wrapper.setRegistrationParameters(MultiSlicePositioner.convertToString(mp.getContext(),parameters));
        wrapper.setTimePoint(0);
        wrapper.setFixedImage(SourcesProcessorHelper.compose(new SourcesZOffset(offset),
                new SourcesChannelsSelect(Arrays.asList(3,4,5))).apply(mp.getReslicedAtlas().nonExtendedSlicedSources));
        wrapper.setMovingImage(SourcesProcessorHelper.compose(new SourcesZOffset(slice),
                new SourcesChannelsSelect(0)).apply(slice.getRegisteredSources()));
        if (!wrapper.register() || !Double.isFinite(measured[0]))
            throw new IllegalStateException("ABBA coordinate calibration failed. A flat coronal session is required.");
        return measured[0];
    }

    void apply(JsonArray updates) {
        if (updates.size() == 0) return;
        checkInterrupted();
        new MarkActionSequenceBatchAction(mp).runRequest();
        try {
            for (JsonElement element : updates) {
                JsonObject update = element.getAsJsonObject();
                String id = update.get("id").getAsString();
                SliceSources slice = slices.get(id);
                if (slice == null) throw new IllegalArgumentException("Unknown section in worker result: " + id);
                int expected = baseline.get(id) + (owned.containsKey(id) ? 1 : 0);
                if (slice.getNumberOfRegistrations() != expected)
                    throw new IllegalStateException("Registration changed outside this run. Stop and restart LangSlice.");
                if (update.has("position_mm")) {
                    double ap = finite(update.get("position_mm"));
                    mp.moveSlice(slice, axisScale * ap + axisOffset);
                }
                boolean changed = update.has("flip") || update.has("rotation_deg") || update.has("affine_mm") || update.has("spline_source_mm");
                if (!changed) continue;
                JsonObject transform = transforms.containsKey(id) ? transforms.get(id).deepCopy() : new JsonObject();
                if (update.has("affine_mm")) {
                    transform.remove("spline_source_mm"); transform.remove("spline_target_mm");
                }
                if (update.has("spline_source_mm")) transform.remove("affine_mm");
                for (Map.Entry<String, JsonElement> value : update.entrySet()) transform.add(value.getKey(), value.getValue());
                Registration<SourceAndConverter<?>[]> next = prepareRegistration(transform);
                Registration<SourceAndConverter<?>[]> previous = owned.get(id);
                if (previous != null) {
                    new DeleteLastRegistrationAction(mp, slice).runRequest();
                    mp.waitForTasks();
                    if (slice.getNumberOfRegistrations()!=baseline.get(id))
                        throw new IllegalStateException("Could not replace the previous correction.");
                }
                try {
                    if (next != null) {
                        appendRegistration(slice,next);
                        if (slice.getNumberOfRegistrations()!=baseline.get(id)+1)
                            throw new IllegalStateException("ABBA did not retain the new correction.");
                        owned.put(id,next);
                    } else owned.remove(id);
                } catch (RuntimeException failure) {
                    // Recover the last completed correction before reporting the failed revision.
                    if (slice.getNumberOfRegistrations()==baseline.get(id)+1) {
                        new DeleteLastRegistrationAction(mp,slice).runRequest();
                        mp.waitForTasks();
                    }
                    if (previous!=null && slice.getNumberOfRegistrations()==baseline.get(id))
                        appendRegistration(slice,previous);
                    throw failure;
                }
                transforms.put(id, transform);
            }
        } finally {
            new MarkActionSequenceBatchAction(mp).runRequest();
            mp.waitForTasks();
        }
    }

    private void appendRegistration(SliceSources slice, Registration<SourceAndConverter<?>[]> registration) {
        RegisterSliceAction action = new RegisterSliceAction(mp,slice,registration,
                SourcesProcessorHelper.Identity(),SourcesProcessorHelper.Identity());
        action.runRequest();
        if (!slice.waitForEndOfAction(action) || !registration.isRegistrationDone())
            throw new IllegalStateException("ABBA could not apply the registration for " + slice.getName());
    }

    static double finite(JsonElement value) {
        double number = value.getAsDouble();
        if (!Double.isFinite(number)) throw new IllegalArgumentException("Non-finite registration coordinate.");
        return number;
    }

    static AffineTransform3D orientation(JsonObject transform) {
        double rotation = transform.has("rotation_deg") ? finite(transform.get("rotation_deg")) : 0;
        if (rotation % 90 != 0) throw new IllegalArgumentException("Orientation must be a quarter turn.");
        AffineTransform3D matrix = new AffineTransform3D();
        matrix.rotate(2, Math.toRadians(-rotation));
        if (transform.has("flip") && transform.get("flip").getAsBoolean()) matrix.scale(-1, 1, 1);
        return matrix;
    }

    private Registration<SourceAndConverter<?>[]> prepareRegistration(JsonObject transform) {
        AffineTransform3D orient = orientation(transform);
        PluginService service = mp.getContext().getService(PluginService.class);
        if (transform.has("spline_source_mm")) {
            double[][] source = points(transform.getAsJsonArray("spline_source_mm"));
            double[][] target = points(transform.getAsJsonArray("spline_target_mm"));
            if (source[0].length != target[0].length) throw new IllegalArgumentException("Unpaired spline points.");
            InvertibleRealTransformSequence pullback = new InvertibleRealTransformSequence();
            pullback.add(tps(target, source));
            pullback.add(orient.inverse());
            SacBigWarp2DRegistration reg = instance(service, SacBigWarp2DRegistration.class);
            reg.setRealTransform(pullback);
            reg.setTransform(reg.getTransform());
            reg.setRegistrationParameters(new HashMap<>());
            reg.setRegistrationName("LangSlice agent");
            if (!reg.isRegistrationDone()) throw new IllegalStateException("Spline serialization failed.");
            return reg;
        }
        AffineTransform3D affine = new AffineTransform3D();
        if (transform.has("affine_mm") && !transform.get("affine_mm").isJsonNull()) {
            JsonArray rows = transform.getAsJsonArray("affine_mm");
            if (rows.size() != 3) throw new IllegalArgumentException("Invalid affine dimensions.");
            for (int r=0; r<3; r++) {
                JsonArray row = rows.get(r).getAsJsonArray();
                if (row.size() != 4) throw new IllegalArgumentException("Invalid affine dimensions.");
                for (int c=0; c<4; c++) affine.set(finite(row.get(c)), r, c);
            }
        }
        double determinant = affine.get(0,0)*affine.get(1,1)-affine.get(0,1)*affine.get(1,0);
        if (!Double.isFinite(determinant) || Math.abs(determinant)<1e-12)
            throw new IllegalArgumentException("Affine registration is singular.");
        affine.concatenate(orient);
        if (affine.isIdentity()) return null;
        AffineRegistration reg = instance(service, AffineRegistration.class);
        Map<String,Object> parameters = new HashMap<>();
        parameters.put("transform", AffineRegistration.affineTransform3DToString(affine));
        parameters.put("pz", 0);
        reg.setRegistrationParameters(MultiSlicePositioner.convertToString(mp.getContext(), parameters));
        reg.setRegistrationName("LangSlice agent");
        return reg;
    }

    private <T extends Registration<SourceAndConverter<?>[]> & org.scijava.plugin.SciJavaPlugin> T instance(PluginService service, Class<T> type) {
        try {
            T reg = type.cast(service.getPlugin(type).createInstance());
            reg.setScijavaContext(mp.getContext());
            return reg;
        } catch (Exception e) { throw new IllegalStateException("ABBA registration plugin is unavailable.", e); }
    }

    static double[][] points(JsonArray rows) {
        if (rows.size()<3) throw new IllegalArgumentException("At least three paired points are required.");
        double[][] output = new double[2][rows.size()];
        for (int i=0;i<rows.size();i++) {
            JsonArray point = rows.get(i).getAsJsonArray();
            if (point.size()!=2) throw new IllegalArgumentException("Expected XY points.");
            output[0][i]=finite(point.get(0)); output[1][i]=finite(point.get(1));
        }
        return output;
    }

    static InvertibleRealTransform tps(double[][] source, double[][] target) {
        ThinplateSplineTransform spline = new ThinplateSplineTransform(source,target);
        if (!interpolates(spline,source,target)) spline = new ThinplateSplineTransform(target,source);
        if (!interpolates(spline,source,target)) throw new IllegalArgumentException("Spline does not reproduce its paired coordinates.");
        return new InvertibleWrapped2DTransformAs3D(new WrappedIterativeInvertibleRealTransform<>(spline));
    }

    private static boolean interpolates(ThinplateSplineTransform spline,double[][] source,double[][] target) {
        double[] sample = new double[2];
        for (int i=0;i<source[0].length;i++) {
            spline.apply(new double[]{source[0][i],source[1][i]},sample);
            double error=Math.hypot(sample[0]-target[0][i],sample[1]-target[1][i]);
            if (!Double.isFinite(error) || error>1e-3) return false;
        }
        return true;
    }

    static void checkInterrupted() {
        if (Thread.currentThread().isInterrupted()) throw new java.util.concurrent.CancellationException("Run cancelled.");
    }
}
