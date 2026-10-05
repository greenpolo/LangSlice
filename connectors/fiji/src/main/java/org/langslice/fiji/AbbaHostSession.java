package org.langslice.fiji;

import bdv.viewer.SourceAndConverter;
import ch.epfl.biop.atlas.aligner.*;
import ch.epfl.biop.atlas.aligner.action.MarkActionSequenceBatchAction;
import ch.epfl.biop.registration.Registration;
import ch.epfl.biop.registration.source.mirror.MirrorXRegistration;
import ch.epfl.biop.registration.source.affine.AffineRegistration;
import ch.epfl.biop.registration.source.bigwarp.BigWarpSource2DRegistration;
import ch.epfl.biop.registration.source.spline.RealTransformSourceRegistration;
import ch.epfl.biop.source.processor.SourcesChannelsSelect;
import ch.epfl.biop.source.processor.SourcesProcessorHelper;
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

/**
 * Calibrated snapshots and native, undoable registrations in an existing ABBA 0.24 session.
 *
 * <p>Each section may carry at most two LangSlice steps, on top of whatever registrations it had
 * when the run started: LangSlice's affine step and, on top of it, LangSlice's warp step (a BigWarp
 * thin-plate spline, ABBA's own registration type, so a saved project reopens and edits without
 * LangSlice). A replacement deletes the newest steps and appends new ones; that is only done while
 * LangSlice's steps are still the newest of the section. Every checkpoint is one ABBA undo step.
 */
final class AbbaHostSession {
    static final String AFFINE_NAME = "LangSlice affine";
    static final String WARP_NAME = "LangSlice warp";
    /** Steps whose name starts with this belong to LangSlice (this run or an earlier one). */
    static final String OWNED_PREFIX = "LangSlice";

    final MultiSlicePositioner mp;
    final Path folder;
    final LinkedHashMap<String, SliceSources> slices = new LinkedHashMap<>();
    private final Map<String, Integer> baseline = new HashMap<>();
    private final Map<String, Registration<SourceAndConverter<?>[]>> ownedAffine = new HashMap<>(), ownedWarp = new HashMap<>();
    private final Map<String, JsonObject> transforms = new HashMap<>();
    /** Rows that could not be applied yet, merged into the next checkpoint and retried. */
    private final LinkedHashMap<String, JsonObject> pending = new LinkedHashMap<>();
    private final Map<String, String> pendingReasons = new LinkedHashMap<>();
    private JsonObject pendingAngles;
    /** Whether the current checkpoint has opened its ABBA batch (opened lazily, at its first real change). */
    private boolean batchOpen;

    AbbaHostSession(MultiSlicePositioner mp, Path folder) {
        this.mp = mp;
        this.folder = folder;
    }

    /**
     * Exports the listed slices as calibrated snapshots and builds the linear.run request.
     * Each snapshot has one page per exported channel, in the order given. Nothing in ABBA changes.
     *
     * @param channelNames every channel's name (all channels, not only the exported ones), or empty
     */
    JsonObject prepare(JsonObject spec, List<SliceSources> chosen, List<Integer> channels, List<String> channelNames,
            double pixelSize, Map<SliceSources, String> damaged, boolean lockRegistered) throws IOException {
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
        double[] half = frame(pixelSize);
        Files.createDirectories(folder);
        JsonObject positions = new JsonObject();
        JsonArray locked = new JsonArray(), warped = new JsonArray();
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
                // Their snapshot already carries the registration: the agent keeps its in-plane geometry.
                if (lockRegistered) locked.add(name);
            }
            if (hasForeignWarp(slice)) warped.add(name);
            if (damaged.containsKey(slice)) marked.addProperty(name, damaged.get(slice) == null ? "" : damaged.get(slice));
            positions.addProperty(name, mp.toAtlasZ(slice.getSlicingAxisPosition()));
            mapping.addProperty(name, slice.getName());
        }
        Files.writeString(folder.resolve("abba_sections.json"), new GsonBuilder().setPrettyPrinting().create().toJson(mapping));
        JsonObject request = new JsonObject();
        request.addProperty("image_folder", folder.toString());
        request.addProperty("pixel_size_um", pixelSize);
        request.add("positions_mm", positions);
        request.addProperty("z_offset_mm", mp.getReslicedAtlas().getZOffset());
        request.add("angles_deg", anglesDeg(mp.getReslicedAtlas().getRotateX(), mp.getReslicedAtlas().getRotateY()));
        request.add("locked", locked);
        request.add("existing_warp", warped);
        request.add("damaged", marked);
        request.add("channel_names", exportedNames(channels, channelNames));
        request.add("spec", spec);
        return request;
    }

    /** The exported pages' names, in page order: never empty, never repeated (the worker refuses duplicates). */
    static JsonArray exportedNames(List<Integer> channels, List<String> names) {
        JsonArray out = new JsonArray();
        Set<String> used = new HashSet<>();
        for (int page = 0; page < channels.size(); page++) {
            int channel = channels.get(page);
            String name = channel < names.size() && names.get(channel) != null ? names.get(channel).trim() : "";
            if (name.isEmpty()) name = "ch" + (channel + 1);
            String unique = name;
            for (int n = 2; !used.add(unique); n++) unique = name + " (" + n + ")";
            out.add(unique);
        }
        return out;
    }

    /** ABBA's rotations (radians) as the job's stack-wide angles; signs pinned by tests/test_abba_angles.py. */
    static JsonObject anglesDeg(double rotateX, double rotateY) {
        JsonObject angles = new JsonObject();
        angles.addProperty("pitch_deg", -Math.toDegrees(rotateX));
        angles.addProperty("yaw_deg", -Math.toDegrees(rotateY));
        return angles;
    }

    /** {rotateX, rotateY} in radians for a host_angles row: rotateX = -radians(pitch), rotateY = -radians(yaw). */
    static double[] rotations(JsonObject angles) {
        double pitch = angles.has("pitch_deg") ? finite(angles.get("pitch_deg")) : 0;
        double yaw = angles.has("yaw_deg") ? finite(angles.get("yaw_deg")) : 0;
        return new double[]{-Math.toRadians(pitch), -Math.toRadians(yaw)};
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
        if (!Double.isFinite(x + y)) throw new IllegalArgumentException("ABBA's atlas cutting angles are not finite.");
    }

    // ---- what a section already carries ---------------------------------------------------------

    /** The section's registrations still in effect, oldest first, compiled from its actions as ABBA's Remove Last does. */
    List<Registration<SourceAndConverter<?>[]>> activeRegistrations(SliceSources slice) {
        List<CancelableAction> actions = mp.getActionsFromSlice(slice);
        List<Registration<SourceAndConverter<?>[]>> active = new ArrayList<>();
        if (actions == null) return active;
        for (CancelableAction action : new ArrayList<>(actions)) {
            if (action instanceof RegisterSliceAction) {
                Registration<SourceAndConverter<?>[]> registration = ((RegisterSliceAction) action).getRegistration();
                if (action.isValid() && registration != null) active.add(registration);
            } else if (action instanceof DeleteLastRegistrationAction && action.isValid() && !active.isEmpty())
                active.remove(active.size() - 1);
        }
        return active;
    }

    /** A spline / BigWarp step that is not LangSlice's own: ABBA's only nonlinear registrations. */
    static boolean isWarp(Registration<?> registration) {
        if (registration instanceof MirrorXRegistration) return false;
        if (registration instanceof RealTransformSourceRegistration) return true;
        String type = registration.getRegistrationTypeName();
        return type != null && (type.contains("Spline") || type.contains("BigWarp"));
    }

    static boolean isOwned(Registration<?> registration) {
        String name = registration.getRegistrationName();
        return name != null && name.startsWith(OWNED_PREFIX);
    }

    boolean hasForeignWarp(SliceSources slice) {
        for (Registration<SourceAndConverter<?>[]> registration : activeRegistrations(slice))
            if (isWarp(registration) && !isOwned(registration)) return true;
        return false;
    }

    // ---- live application -----------------------------------------------------------------------

    /** What one application did: rows applied and rows kept for the next attempt, with the reason. */
    static final class ApplyReport {
        final List<String> applied = new ArrayList<>();
        final LinkedHashMap<String, String> failed = new LinkedHashMap<>();
        boolean angles, anglesFailed;
        String anglesReason;

        boolean changedAnything() { return !applied.isEmpty() || angles; }

        String summary() {
            StringBuilder text = new StringBuilder();
            if (anglesFailed) text.append("Cutting angles not applied yet: ").append(anglesReason).append(' ');
            failed.forEach((id, reason) -> text.append(id).append(": ").append(reason).append(' '));
            return text.toString().trim();
        }

        JsonObject toJson() {
            JsonObject json = new JsonObject();
            JsonArray done = new JsonArray(); applied.forEach(done::add);
            JsonObject kept = new JsonObject(); failed.forEach(kept::addProperty);
            json.add("applied", done); json.add("failed", kept);
            json.addProperty("angles", angles);
            if (anglesFailed) json.addProperty("angles_failed", anglesReason);
            return json;
        }
    }

    boolean hasPending() { return !pending.isEmpty() || pendingAngles != null; }

    /** Rows kept after a failure, and why, for the run window. */
    synchronized Map<String, String> pendingReasons() { return new LinkedHashMap<>(pendingReasons); }

    /**
     * Applies one checkpoint live, as ONE ABBA undo step: its stack-wide cutting angles (when present),
     * then every row (position, affine step, warp step). Rows kept from an earlier failed attempt are merged
     * in first and retried. A row that fails is reported and kept; it never stops the others.
     */
    synchronized ApplyReport applyCheckpoint(JsonArray updates, JsonObject hostAngles) {
        ApplyReport report = new ApplyReport();
        LinkedHashMap<String, JsonObject> rows = new LinkedHashMap<>();
        pending.forEach((id, row) -> rows.put(id, row.deepCopy()));
        if (updates != null) for (JsonElement element : updates) {
            JsonObject update = element.getAsJsonObject();
            String id = update.get("id").getAsString();
            rows.put(id, merge(rows.get(id), update));
        }
        JsonObject angles = hostAngles != null ? hostAngles : pendingAngles;
        if (rows.isEmpty() && angles == null) return report;
        pending.clear(); pendingReasons.clear(); pendingAngles = null;
        mp.waitForTasks();
        batchOpen = false;
        try {
            if (angles != null) {
                try { applyAngles(angles); report.angles = true; }
                catch (RuntimeException failure) {
                    report.anglesFailed = true; report.anglesReason = reason(failure); pendingAngles = angles;
                }
            }
            for (Map.Entry<String, JsonObject> row : rows.entrySet()) {
                try { applyRow(row.getKey(), row.getValue()); report.applied.add(row.getKey()); }
                catch (RuntimeException failure) {
                    String why = reason(failure);
                    report.failed.put(row.getKey(), why);
                    pending.put(row.getKey(), row.getValue());
                    pendingReasons.put(row.getKey(), why);
                }
            }
        } finally {
            // Nothing changed: no marks at all, so ABBA's Redo history is left as it was.
            if (batchOpen) new MarkActionSequenceBatchAction(mp).runRequest();
            batchOpen = false;
            mp.waitForTasks();
        }
        return report;
    }

    /** One older row with a newer one on top: the newer keys win. */
    static JsonObject merge(JsonObject older, JsonObject newer) {
        if (older == null) return newer.deepCopy();
        JsonObject merged = older.deepCopy();
        for (Map.Entry<String, JsonElement> value : newer.entrySet()) merged.add(value.getKey(), value.getValue().deepCopy());
        return merged;
    }

    private static String reason(RuntimeException failure) {
        return failure.getMessage() == null ? failure.getClass().getSimpleName() : failure.getMessage();
    }

    /** Opens the checkpoint's undo batch right before its first change in ABBA. */
    private void begin() {
        if (batchOpen) return;
        new MarkActionSequenceBatchAction(mp).runRequest();
        batchOpen = true;
    }

    private void applyAngles(JsonObject angles) {
        double[] rotation = rotations(angles);
        begin();
        new SlicingAnglesAction(mp, rotation[0], rotation[1]).runRequest();
        ReslicedAtlas atlas = mp.getReslicedAtlas();
        if (Math.abs(atlas.getRotateX() - rotation[0]) > 1e-9 || Math.abs(atlas.getRotateY() - rotation[1]) > 1e-9)
            throw new IllegalStateException("ABBA's atlas slicing is locked; the cutting angles were not changed.");
    }

    private void applyRow(String id, JsonObject update) {
        SliceSources slice = slices.get(id);
        if (slice == null) throw new IllegalArgumentException("Unknown section in the worker's result.");
        if (!mp.getSlices().contains(slice)) throw new IllegalStateException("The slice was removed from ABBA.");
        boolean placement = update.has("flip") || update.has("rotation_deg") || update.has("affine_mm");
        boolean warpChange = update.has("warp");
        Registration<SourceAndConverter<?>[]> previousAffine = ownedAffine.get(id), previousWarp = ownedWarp.get(id);
        if (placement || warpChange) {
            slice.waitForEndOfTasks();
            requireOwnStepsNewest(id, slice);
        }
        // Build every new step before anything in ABBA changes.
        JsonObject transform = transforms.containsKey(id) ? transforms.get(id).deepCopy() : new JsonObject();
        Registration<SourceAndConverter<?>[]> nextAffine = previousAffine, nextWarp = previousWarp;
        if (placement) {
            for (String key : new String[]{"flip", "rotation_deg", "affine_mm"})
                if (update.has(key)) transform.add(key, update.get(key).deepCopy());
            nextAffine = prepareRegistration(transform);
        }
        if (warpChange) nextWarp = update.get("warp").isJsonNull() ? null : prepareWarp(update.getAsJsonObject("warp"));

        if (update.has("position_mm")) { begin(); mp.moveSlice(slice, mp.fromAtlasZ(finite(update.get("position_mm")))); }
        if (!placement && !warpChange) return;
        begin();

        // Remove from the top: the warp first, then (only when the placement changes) the affine.
        int base = baseline.get(id);
        if (previousWarp != null) deleteLast(slice);
        if (placement && previousAffine != null) deleteLast(slice);
        int expected = base + (!placement && previousAffine != null ? 1 : 0);
        if (slice.getNumberOfRegistrations() != expected)
            throw new IllegalStateException("Could not remove LangSlice's previous steps; use ABBA's Undo to restore them.");
        try {
            if (placement && nextAffine != null) appendRegistration(slice, nextAffine);
            if (nextWarp != null) appendRegistration(slice, nextWarp);
        } catch (RuntimeException failure) {
            // Put back the steps that were there before this row, then report the row as failed.
            while (slice.getNumberOfRegistrations() > expected) deleteLast(slice);
            if (placement && previousAffine != null) appendRegistration(slice, previousAffine);
            if (previousWarp != null) appendRegistration(slice, previousWarp);
            throw failure;
        }
        if (placement) {
            if (nextAffine != null) ownedAffine.put(id, nextAffine); else ownedAffine.remove(id);
            transforms.put(id, transform);
        }
        if (nextWarp != null) ownedWarp.put(id, nextWarp); else ownedWarp.remove(id);
    }

    /** LangSlice may replace its steps only while they are the section's newest; anything else was changed in ABBA. */
    private void requireOwnStepsNewest(String id, SliceSources slice) {
        List<Registration<SourceAndConverter<?>[]>> owned = new ArrayList<>();
        if (ownedAffine.containsKey(id)) owned.add(ownedAffine.get(id));
        if (ownedWarp.containsKey(id)) owned.add(ownedWarp.get(id));
        boolean counted = slice.getNumberOfRegistrations() == baseline.get(id) + owned.size();
        List<Registration<SourceAndConverter<?>[]>> active = activeRegistrations(slice);
        boolean newest = owned.isEmpty();
        if (!newest && active.size() == slice.getNumberOfRegistrations())
            // The same objects LangSlice appended, on top (identity, not equality of transforms).
            newest = active.size() >= owned.size() && active.subList(active.size() - owned.size(), active.size()).equals(owned);
        else if (!newest) newest = counted; // ABBA's action list disagrees with its count: rely on the count alone.
        if (!counted || !newest)
            throw new IllegalStateException("Its registrations were changed in ABBA during the run, so LangSlice's steps are no longer "
                    + "the newest; LangSlice leaves this slice alone.");
    }

    private void deleteLast(SliceSources slice) {
        new DeleteLastRegistrationAction(mp, slice).runRequest();
        slice.waitForEndOfTasks();
        mp.waitForTasks();
    }

    private void appendRegistration(SliceSources slice, Registration<SourceAndConverter<?>[]> registration) {
        int before = slice.getNumberOfRegistrations();
        RegisterSliceAction action = new RegisterSliceAction(mp, slice, registration,
                SourcesProcessorHelper.Identity(), SourcesProcessorHelper.Identity());
        action.runRequest();
        if (!slice.waitForEndOfAction(action) || !registration.isRegistrationDone() || slice.getNumberOfRegistrations() != before + 1)
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

    /** The affine step, or null for the identity. */
    Registration<SourceAndConverter<?>[]> prepareRegistration(JsonObject transform) {
        PluginService service = mp.getContext().getService(PluginService.class);
        AffineTransform3D affine = affineMatrix(transform);
        if (affine == null) return null;
        AffineRegistration reg = instance(service, AffineRegistration.class);
        Map<String, Object> parameters = new HashMap<>();
        parameters.put(AffineRegistration.TRANSFORM_KEY, AffineRegistration.affineTransform3DToString(affine));
        parameters.put("pz", 0);
        reg.setRegistrationParameters(MultiSlicePositioner.convertToString(mp.getContext(), parameters));
        reg.setRegistrationName(AFFINE_NAME);
        if (!reg.register()) throw new IllegalStateException("ABBA could not read the affine registration.");
        return reg;
    }

    /** affine_mm with the orientation applied first; null when the result is the identity. */
    static AffineTransform3D affineMatrix(JsonObject transform) {
        AffineTransform3D affine = new AffineTransform3D();
        if (transform.has("affine_mm") && !transform.get("affine_mm").isJsonNull()) {
            JsonArray rows = transform.getAsJsonArray("affine_mm");
            if (rows.size() != 3) throw new IllegalArgumentException("Invalid affine dimensions.");
            for (int r = 0; r < 3; r++) {
                JsonArray row = rows.get(r).getAsJsonArray();
                if (row.size() != 4) throw new IllegalArgumentException("Invalid affine dimensions.");
                for (int c = 0; c < 4; c++) affine.set(finite(row.get(c)), r, c);
            }
        }
        double determinant = affine.get(0, 0) * affine.get(1, 1) - affine.get(0, 1) * affine.get(1, 0);
        if (!Double.isFinite(determinant) || Math.abs(determinant) < 1e-12)
            throw new IllegalArgumentException("Affine registration is singular.");
        affine.concatenate(orientation(transform));
        return affine.isIdentity() ? null : affine;
    }

    /**
     * The warp step: a BigWarp thin-plate spline from the row's landmark pairs, in the same centred ABBA
     * world-mm frame as the affine step, applied AFTER it.
     * The plain wrapped TPS (no orientation inside) is what BigWarp reopens for editing.
     */
    BigWarpSource2DRegistration prepareWarp(JsonObject warp) {
        BigWarpSource2DRegistration reg = instance(mp.getContext().getService(PluginService.class), BigWarpSource2DRegistration.class);
        reg.setRealTransform(warpTransform(warp));
        reg.setTransform(reg.getTransform());
        Map<String, String> parameters = new HashMap<>();
        if (warp.has("record") && warp.get("record").isJsonPrimitive()) parameters.put("langslice_record", warp.get("record").getAsString());
        if (warp.has("max_error_mm") && warp.get("max_error_mm").isJsonPrimitive()) parameters.put("langslice_max_error_mm", warp.get("max_error_mm").getAsString());
        if (warp.has("p99_error_mm") && warp.get("p99_error_mm").isJsonPrimitive()) parameters.put("langslice_p99_error_mm", warp.get("p99_error_mm").getAsString());
        reg.setRegistrationParameters(parameters);
        reg.setRegistrationName(WARP_NAME);
        if (!reg.isRegistrationDone()) throw new IllegalStateException("Warp serialization failed.");
        return reg;
    }

    /** source_mm / target_mm as a TPS mapping target points to source points (a pull-back, as BigWarp stores it). */
    static InvertibleRealTransform warpTransform(JsonObject warp) {
        if (!warp.has("source_mm") || !warp.has("target_mm")) throw new IllegalArgumentException("A warp needs source_mm and target_mm.");
        double[][] source = coordinates(warp.getAsJsonArray("source_mm"));
        double[][] target = coordinates(warp.getAsJsonArray("target_mm"));
        if (source[0].length != target[0].length) throw new IllegalArgumentException("Unpaired warp points.");
        if (warp.has("points") && warp.get("points").isJsonPrimitive() && warp.get("points").getAsInt() != source[0].length)
            throw new IllegalArgumentException("The warp's point count does not match its coordinates.");
        return tps(target, source);
    }

    /** XY coordinates as [[x...],[y...]] (two rows) or as [[x, y], ...] (one pair per point); at least three points. */
    static double[][] coordinates(JsonArray value) {
        boolean rows = value.size() == 2 && value.get(0).isJsonArray() && value.get(0).getAsJsonArray().size() >= 3;
        if (!rows) return points(value);
        JsonArray xs = value.get(0).getAsJsonArray(), ys = value.get(1).getAsJsonArray();
        if (xs.size() != ys.size()) throw new IllegalArgumentException("Warp coordinate rows differ in length.");
        double[][] output = new double[2][xs.size()];
        for (int i = 0; i < xs.size(); i++) { output[0][i] = finite(xs.get(i)); output[1][i] = finite(ys.get(i)); }
        return output;
    }

    private <T extends Registration<SourceAndConverter<?>[]> & org.scijava.plugin.SciJavaPlugin> T instance(PluginService service, Class<T> type) {
        try {
            T reg = type.cast(service.getPlugin(type).createInstance());
            reg.setScijavaContext(mp.getContext());
            return reg;
        } catch (Exception e) { throw new IllegalStateException("ABBA registration plugin is unavailable.", e); }
    }

    static double[][] points(JsonArray rows) {
        if (rows.size() < 3) throw new IllegalArgumentException("At least three paired points are required.");
        double[][] output = new double[2][rows.size()];
        for (int i = 0; i < rows.size(); i++) {
            JsonArray point = rows.get(i).getAsJsonArray();
            if (point.size() != 2) throw new IllegalArgumentException("Expected XY points.");
            output[0][i] = finite(point.get(0)); output[1][i] = finite(point.get(1));
        }
        return output;
    }

    static InvertibleRealTransform tps(double[][] source, double[][] target) {
        ThinplateSplineTransform spline = new ThinplateSplineTransform(source, target);
        if (!interpolates(spline, source, target)) spline = new ThinplateSplineTransform(target, source);
        if (!interpolates(spline, source, target)) throw new IllegalArgumentException("Spline does not reproduce its paired coordinates.");
        return new InvertibleWrapped2DTransformAs3D(new WrappedIterativeInvertibleRealTransform<>(spline));
    }

    private static boolean interpolates(ThinplateSplineTransform spline, double[][] source, double[][] target) {
        double[] sample = new double[2];
        for (int i = 0; i < source[0].length; i++) {
            spline.apply(new double[]{source[0][i], source[1][i]}, sample);
            double error = Math.hypot(sample[0] - target[0][i], sample[1] - target[1][i]);
            if (!Double.isFinite(error) || error > 1e-3) return false;
        }
        return true;
    }

    static void checkInterrupted() {
        if (Thread.currentThread().isInterrupted()) throw new java.util.concurrent.CancellationException("Run cancelled.");
    }
}
