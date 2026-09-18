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
import ij.io.FileSaver;
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

    JsonObject prepare(JsonObject spec, int channel, double pixelSize) throws IOException {
        validateSession();
        if (!Double.isFinite(pixelSize) || pixelSize <= 0 || channel < 0)
            throw new IllegalArgumentException("Choose a positive pixel size and a valid channel.");
        mp.waitForTasks();
        List<SliceSources> selected = new ArrayList<>(mp.getSelectedSlices());
        if (selected.isEmpty()) selected.addAll(mp.getSlices());
        selected.sort(Comparator.comparingDouble(SliceSources::getSlicingAxisPosition));
        if (selected.isEmpty()) throw new IllegalArgumentException("Import sections before running LangSlice.");
        boolean reorder = spec.getAsJsonArray("tasks").contains(new JsonPrimitive("reorder"));
        if (reorder && selected.stream().anyMatch(s -> s.getNumberOfRegistrations() > 0))
            throw new IllegalArgumentException("Turn off ordering for sections with existing registrations.");
        for (SliceSources slice : selected)
            if (channel >= slice.getRegisteredSources().length)
                throw new IllegalArgumentException("The selected channel is missing from " + slice.getName());
        measureAxis(selected.get(0));
        double[] roi = mp.getROI();
        double spacing = pixelSize / 1000.0;
        double halfX = Math.ceil(Math.max(Math.abs(roi[0]), Math.abs(roi[0] + roi[2])) / spacing) * spacing;
        double halfY = Math.ceil(Math.max(Math.abs(roi[1]), Math.abs(roi[1] + roi[3])) / spacing) * spacing;
        if (!Double.isFinite(halfX + halfY) || halfX <= 0 || halfY <= 0)
            throw new IllegalArgumentException("ABBA's image region has no usable extent.");
        Files.createDirectories(folder);
        JsonObject positions = new JsonObject();
        JsonArray registered = new JsonArray();
        JsonObject mapping = new JsonObject();
        for (int index = 0; index < selected.size(); index++) {
            checkInterrupted();
            SliceSources slice = selected.get(index);
            String name = String.format(Locale.ROOT, "section_%04d.tif", index + 1);
            ImagePlus image = SliceToImagePlus.export(slice, new SourcesChannelsSelect(channel),
                    -halfX, -halfY, 2 * halfX, 2 * halfY, spacing, 0, true);
            try {
                if (!new FileSaver(image).saveAsTiff(folder.resolve(name).toString()))
                    throw new IOException("Could not save section snapshot " + name);
            } finally { image.close(); }
            slices.put(name, slice);
            baseline.put(name, slice.getNumberOfRegistrations());
            if (slice.getNumberOfRegistrations() > 0) registered.add(name);
            positions.addProperty(name, (slice.getSlicingAxisPosition() - axisOffset) / axisScale);
            mapping.addProperty(name, slice.getName());
        }
        Files.writeString(folder.resolve("abba_sections.json"), new GsonBuilder().setPrettyPrinting().create().toJson(mapping));
        JsonObject request = new JsonObject();
        request.addProperty("image_folder", folder.toString());
        request.addProperty("pixel_size_um", pixelSize);
        request.add("positions_mm", positions);
        request.add("registered_slices", registered);
        request.add("spec", spec);
        return request;
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

    /** Prepare the SimpleRegistrationWrapper off-stack, then save only a native BigWarp step. */
    void nonlinear(Path environment, JsonObject config, int channel, double pixelSize,
            java.util.function.Consumer<JsonObject> events,
            java.util.function.Consumer<WorkerClient> clientChanged,
            java.util.concurrent.atomic.AtomicBoolean stopping) throws Exception {
        validateSession();
        if (!Double.isFinite(pixelSize) || pixelSize<=0) throw new IllegalArgumentException("Choose a positive pixel size.");
        mp.waitForTasks();
        List<SliceSources> selected = new ArrayList<>(mp.getSelectedSlices());
        if (selected.isEmpty()) selected.addAll(mp.getSlices());
        if (selected.isEmpty()) throw new IllegalArgumentException("Import sections before running LangSlice.");
        Files.createDirectories(folder);
        for (SliceSources slice : selected) {
            if (stopping.get()) throw new java.util.concurrent.CancellationException();
            checkInterrupted();
            int before = slice.getNumberOfRegistrations();
            if (channel<0 || channel>=slice.getRegisteredSources().length)
                throw new IllegalArgumentException("The selected image channel is missing.");
            SimpleRegistrationPlugin plugin = new SimpleRegistrationPlugin() {
                public double getVoxelSizeInMicron() { return pixelSize; }
                public void setRegistrationParameters(Map<String,String> parameters) { }
                public InvertibleRealTransform register(ImagePlus fixed, ImagePlus moving, ImagePlus fm, ImagePlus mm) {
                    try {
                        float ap = fixed.getStack().getProcessor(1).getf(fixed.getWidth()/2,fixed.getHeight()/2);
                        float apX = fixed.getStack().getProcessor(1).getf(fixed.getWidth()/3,fixed.getHeight()/2);
                        float apY = fixed.getStack().getProcessor(1).getf(fixed.getWidth()/2,fixed.getHeight()/3);
                        if (!Float.isFinite(ap) || Math.abs(ap-apX)>1e-4 || Math.abs(ap-apY)>1e-4)
                            throw new IllegalArgumentException("Nonlinear registration requires a flat coronal atlas plane.");
                        Path section = Files.createTempDirectory(folder,"section-");
                        Path coordinates = section.resolve("atlas_coordinates.tif");
                        Path histology = section.resolve("histology.tif");
                        if (!new FileSaver(fixed).saveAsTiffStack(coordinates.toString()) ||
                                !new FileSaver(moving).saveAsTiff(histology.toString()))
                            throw new IOException("Could not write calibrated nonlinear inputs.");
                        JsonObject request = new JsonObject();
                        request.addProperty("coords_path",coordinates.toString());
                        request.addProperty("histology_path",histology.toString());
                        request.add("config",config);
                        try (WorkerClient client = new WorkerClient(environment)) {
                            clientChanged.accept(client);
                            if (stopping.get()) throw new java.util.concurrent.CancellationException();
                            JsonObject result = client.request("nonlinear.abba",request,events,java.time.Duration.ofHours(12));
                            if (!"fixed_grid_pixels".equals(result.get("coordinate_frame").getAsString()))
                                throw new IllegalArgumentException("Unsupported nonlinear coordinate frame.");
                            double[][] source = points(result.getAsJsonArray("source_points"));
                            double[][] target = points(result.getAsJsonArray("target_points"));
                            if (source[0].length!=target[0].length) throw new IllegalArgumentException("Unpaired nonlinear points.");
                            return tps(source,target);
                        } finally { clientChanged.accept(null); }
                    } catch (Exception failure) { throw new IllegalStateException("Nonlinear registration failed: " + failure.getMessage(),failure); }
                }
            };
            SimpleRegistrationWrapper wrapper = new SimpleRegistrationWrapper("LangSlice nonlinear preparation",plugin);
            wrapper.setScijavaContext(mp.getContext());
            double[] roi = mp.getROI();
            Map<String,Object> parameters = new HashMap<>();
            parameters.put("px",roi[0]); parameters.put("py",roi[1]);
            parameters.put("sx",roi[2]); parameters.put("sy",roi[3]); parameters.put("pz",0);
            wrapper.setRegistrationParameters(MultiSlicePositioner.convertToString(mp.getContext(),parameters));
            wrapper.setTimePoint(0);
            wrapper.setFixedImage(SourcesProcessorHelper.compose(new SourcesZOffset(slice),
                    new SourcesChannelsSelect(Arrays.asList(3,4,5))).apply(mp.getReslicedAtlas().nonExtendedSlicedSources));
            wrapper.setMovingImage(SourcesProcessorHelper.compose(new SourcesZOffset(slice),
                    new SourcesChannelsSelect(channel)).apply(slice.getRegisteredSources()));
            if (!wrapper.register()) throw new IllegalStateException("ABBA could not prepare the nonlinear correction.");
            if (stopping.get()) throw new java.util.concurrent.CancellationException();
            checkInterrupted();
            if (slice.getNumberOfRegistrations()!=before)
                throw new IllegalStateException("Registration changed outside this run; restart LangSlice.");
            SacBigWarp2DRegistration registration = instance(mp.getContext().getService(PluginService.class),SacBigWarp2DRegistration.class);
            registration.setRealTransform(wrapper.getRealTransform());
            registration.setTransform(registration.getTransform());
            registration.setRegistrationParameters(new HashMap<>());
            registration.setRegistrationName("LangSlice nonlinear");
            appendRegistration(slice,registration);
        }
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
