package org.langslice.fiji;

import bdv.util.BoundedRealTransform;
import bdv.viewer.SourceAndConverter;
import ch.epfl.biop.atlas.aligner.*;
import ch.epfl.biop.registration.Registration;
import ch.epfl.biop.registration.plugin.SimpleRegistrationPlugin;
import ch.epfl.biop.registration.plugin.SimpleRegistrationWrapper;
import ch.epfl.biop.registration.source.bigwarp.BigWarpSource2DRegistration;
import ch.epfl.biop.registration.source.spline.RealTransformSourceRegistration;
import ch.epfl.biop.source.processor.SourcesChannelsSelect;
import ch.epfl.biop.source.processor.SourcesProcessorHelper;
import com.google.gson.*;
import ij.ImagePlus;
import net.imglib2.realtransform.*;

import java.nio.file.*;
import java.util.*;

/**
 * Opt-in integration smoke against a REAL headless ABBA 0.25.0 with the cached Allen V3p1 atlas; never contacts
 * a model provider. After build.py --test, run with the same resolved test classpath:
 * <pre>java -Djava.awt.headless=true --add-opens=java.base/java.lang=ALL-UNNAMED -cp TEST_CLASSES:CLASSES:MAVEN_CLASSPATH
 *     org.langslice.fiji.HostSessionSmokeTest /path/to/cached_atlas</pre>
 * It then starts a second JVM WITHOUT the connector's classes ({@link ReloadCheck}) to reopen the saved project.
 * Requires the Allen V3p1 XML, HDF5 and ontology already downloaded by ABBA (normally ~/cached_atlas).
 * Checks: AP positions from ABBA's own toAtlasZ against ABBA's atlas coordinate images read at the slice, a tilted
 * session's export (angles_deg, existing_warp, registered, the abba block), live checkpoints (position, affine step,
 * warp step, cutting angles), the affine-then-warp replace/remove order, one checkpoint = one Undo (angles included),
 * refusal and retry after an outside change, ABBA's own reason for a refused registration, save then reload without
 * LangSlice with the warp step intact, and attach: this job's steps (by project key) removed and the job applied again
 * in one Undo step, another key's step and the user's own kept, a section with this job's step under another refused.
 */
public final class HostSessionSmokeTest {
    // ImageJ 1.x classes must be patched before any is loaded (ij1-patcher); Fiji's launcher does this itself.
    static { net.imagej.patcher.LegacyInjector.preinit(); }

    static final String S1 = "section_0001.tif", S2 = "section_0002.tif";
    /** This job's project key, and another job's. */
    static final String KEY = "0123456789ab", OTHER = "fedcba987654";

    public static void main(String[] args) throws Exception {
        if (args.length != 1) throw new IllegalArgumentException("Supply an existing ABBA Allen V3p1 atlas cache folder");
        Path cache = Paths.get(args[0]);
        for (String name : new String[]{"mouse_brain_ccfv3p1.xml", "ccf2017-mod65000-border-centered-mm-bc.h5", "1.json"})
            require(Files.isRegularFile(cache.resolve(name)), "Missing cached atlas file: " + name);
        Path output = Files.createTempDirectory("langslice-host-smoke-");
        net.imagej.ImageJ imagej = new net.imagej.ImageJ();
        MultiSlicePositioner mp = start(imagej, cache);
        for (int i = 0; i < 2; i++) {
            ImagePlus image = ij.IJ.createImage("Synthetic section " + i, "8-bit ramp", 128, 96, 1);
            image.getCalibration().pixelWidth = 25; image.getCalibration().pixelHeight = 25; image.getCalibration().setUnit("um");
            new ij.io.FileSaver(image).saveAsTiff(output.resolve("synthetic" + i + ".tif").toString()); image.close();
        }
        imagej.command().run(ch.epfl.biop.atlas.aligner.command.ImportSlicesFromFilesCommand.class, true,
                "mp", mp, "datasetname", "synthetic",
                "files", new java.io.File[]{output.resolve("synthetic0.tif").toFile(), output.resolve("synthetic1.tif").toFile()},
                "split_rgb_channels", false, "first_slice_position_mm", 5.0, "slice_spacing_mm", .4).get();
        mp.waitForTasks();
        JsonObject expected = run(mp, output);
        Files.writeString(output.resolve("expected.json"), expected.toString());
        imagej.context().dispose();
        reload(cache, output);
        System.out.println("Synthetic run files: " + output);
        System.exit(0);
    }

    static MultiSlicePositioner start(net.imagej.ImageJ imagej, Path cache) throws Exception {
        ch.epfl.biop.atlas.AtlasLocationHelper.defaultCacheDir = cache.toFile();
        Object atlas = imagej.command().run(ch.epfl.biop.atlas.mouse.allen.ccfv3p1.command.AllenBrainAdultMouseAtlasCCF2017v3p1Command.class, true)
                .get().getOutput("ba");
        return (MultiSlicePositioner) imagej.command().run(ch.epfl.biop.atlas.aligner.command.ABBAStartCommand.class, true,
                "x_axis", "RL (Right-Left)", "y_axis", "SI (Superior-Inferior)", "z_axis", "AP (Anterior-Posterior)", "ba", atlas).get().getOutput("mp");
    }

    static void require(boolean condition, String message) { if (!condition) throw new AssertionError(message); }

    static JsonArray rows(JsonObject... rows) { JsonArray array = new JsonArray(); for (JsonObject row : rows) array.add(row); return array; }

    static JsonObject row(String id) { JsonObject row = new JsonObject(); row.addProperty("id", id); return row; }

    static JsonArray affine(double tx, double ty, double scale) {
        return JsonParser.parseString("[[" + scale + ",0.1,0," + tx + "],[-0.05," + scale + ",0," + ty + "],[0,0,1,0]]").getAsJsonArray();
    }

    /** A smooth 5x5 landmark warp, centred ABBA mm. */
    static double[][][] landmarks(double amplitude) {
        double[][] source = new double[2][25], target = new double[2][25];
        for (int i = 0; i < 25; i++) {
            double x = (i % 5 - 2) * .6, y = (i / 5 - 2) * .5;
            source[0][i] = x; source[1][i] = y;
            target[0][i] = x + amplitude * Math.sin(y + .2); target[1][i] = y + amplitude * .7 * Math.cos(x + .3);
        }
        return new double[][][]{source, target};
    }

    static JsonObject warp(double amplitude) {
        double[][][] pairs = landmarks(amplitude);
        return HostGeometrySmokeTest.warpRow(pairs[0], pairs[1]);
    }

    static JsonObject angles(double pitch, double yaw) {
        JsonObject angles = new JsonObject(); angles.addProperty("pitch_deg", pitch); angles.addProperty("yaw_deg", yaw); return angles;
    }

    /** The registrations in effect, by name, oldest first (ABBA's own compile of the slice's actions). */
    static List<String> names(AbbaHostSession host, SliceSources slice) {
        List<String> names = new ArrayList<>();
        for (Registration<SourceAndConverter<?>[]> registration : host.activeRegistrations(slice)) names.add(registration.getRegistrationName());
        return names;
    }

    static RealTransform unbox(RealTransform transform) {
        return transform instanceof BoundedRealTransform ? ((BoundedRealTransform) transform).getTransform() : transform;
    }

    /** The newest registration's pull-back, unboxed, maps every target landmark onto its source landmark. */
    static void warpReadsBack(Registration<SourceAndConverter<?>[]> registration, double amplitude, String message) {
        require(registration instanceof BigWarpSource2DRegistration, message + ": the warp is a BigWarp step, got " + registration);
        RealTransform transform = unbox(((RealTransformSourceRegistration) registration).getRealTransform());
        double[][][] pairs = landmarks(amplitude);
        double[] out = new double[3];
        for (int i = 0; i < 25; i++) {
            transform.apply(new double[]{pairs[1][0][i], pairs[1][1][i], 0}, out);
            require(Math.hypot(out[0] - pairs[0][0][i], out[1] - pairs[0][1][i]) < 1e-6, message + ": landmark " + i + " read back " + Arrays.toString(out));
        }
        require(bdv.util.RealTransformHelper.BigWarpFileFromRealTransform(transform) != null, message + ": BigWarp can reopen it for editing");
    }

    static void anglesAre(MultiSlicePositioner mp, double pitch, double yaw, String message) {
        double[] expected = AbbaHostSession.rotations(angles(pitch, yaw));
        require(Math.abs(mp.getReslicedAtlas().getRotateX() - expected[0]) < 1e-9 && Math.abs(mp.getReslicedAtlas().getRotateY() - expected[1]) < 1e-9,
                message + ": rotateX " + mp.getReslicedAtlas().getRotateX() + " rotateY " + mp.getReslicedAtlas().getRotateY());
    }

    public static JsonObject run(MultiSlicePositioner mp, Path output) throws Exception {
        require(mp.getSlices().size() == 2, "Two synthetic sections imported");
        List<SliceSources> sections = new ArrayList<>(mp.getSlices());
        sections.sort(Comparator.comparingDouble(SliceSources::getSlicingAxisPosition));
        SliceSources one = sections.get(0), two = sections.get(1);
        require(mp.getAtlas().getName().contains("V3p1"), "Allen V3p1 atlas: " + mp.getAtlas().getName());

        // 1. ABBA's own AP conversion equals what its atlas coordinate images read at the slice (flat session).
        double probed = probeAP(mp, one), converted = mp.toAtlasZ(one.getSlicingAxisPosition());
        System.out.println("AP_CHECK probe=" + probed + " toAtlasZ=" + converted + " difference=" + (converted - probed)
                + " zOffset=" + mp.getReslicedAtlas().getZOffset() + " slicingAxisPosition=" + one.getSlicingAxisPosition());
        require(Math.abs(converted - 5.0) < 1e-9, "Import at 5 mm is toAtlasZ 5 mm");
        require(Math.abs(probed - converted) < 0.02, "toAtlasZ agrees with the coordinate probe within one 20 um voxel: " + probed + " vs " + converted);

        // 2. A user's own BigWarp step on section two is reported as existing_warp.
        AbbaHostSession foreign = new AbbaHostSession(mp, output.resolve("unused"), OTHER);
        BigWarpSource2DRegistration userWarp = foreign.prepareWarp(warp(.03));
        userWarp.setRegistrationName("Big Warp");
        RegisterSliceAction user = new RegisterSliceAction(mp, two, userWarp, SourcesProcessorHelper.Identity(), SourcesProcessorHelper.Identity());
        user.runRequest(); require(two.waitForEndOfAction(user), "User warp applied");
        mp.waitForTasks();

        // 3. A tilted session is accepted: its angles travel as the job's stack-wide angles.
        mp.getReslicedAtlas().setRotateX(-Math.toRadians(3)); mp.getReslicedAtlas().setRotateY(Math.toRadians(2));
        AbbaHostSession host = new AbbaHostSession(mp, output.resolve("snapshots"), KEY);
        require(host.snapshotPixelUm(sections, Arrays.asList(0)) == 25.0, "25 µm sources are exported at their own 25 µm, not finer");
        JsonObject request = host.export(sections, Arrays.asList(0), Arrays.asList("ramp"), 25.0, 40, 400);
        System.out.println("EXPORT " + request.get("angles_deg") + " z_offset_mm=" + request.get("z_offset_mm")
                + " positions=" + request.get("positions_mm") + " existing_warp=" + request.get("existing_warp") + " abba=" + request.get("abba"));
        require(Math.abs(request.getAsJsonObject("angles_deg").get("pitch_deg").getAsDouble() - 3) < 1e-9
                && Math.abs(request.getAsJsonObject("angles_deg").get("yaw_deg").getAsDouble() + 2) < 1e-9, "angles_deg carries ABBA's tilt");
        require(request.get("z_offset_mm").getAsDouble() == mp.getReslicedAtlas().getZOffset(), "z_offset_mm is ABBA's getZOffset");
        require(Math.abs(request.getAsJsonObject("positions_mm").get(S1).getAsDouble() - 5.0) < 1e-9
                && Math.abs(request.getAsJsonObject("positions_mm").get(S2).getAsDouble() - 5.4) < 1e-9, "positions_mm = toAtlasZ");
        require(request.getAsJsonArray("existing_warp").toString().equals("[\"" + S2 + "\"]"), "existing_warp names the user's warped slice only");
        require(request.getAsJsonArray("registered").toString().equals("[\"" + S2 + "\"]"), "registered names the slices that carry registrations");
        require(!request.has("spec") && !request.has("damaged") && !request.has("preprocessing"), "No spec, damage or preprocessing: the generator sets them");
        require(request.getAsJsonArray("channel_names").toString().equals("[\"ramp\"]"), "channel_names sent");
        JsonObject abba = request.getAsJsonObject("abba");
        require(abba.get("project_key").getAsString().equals(KEY) && abba.get("atlas").getAsString().equals(mp.getAtlas().getName())
                && abba.getAsJsonObject("sections").get(S1).getAsString().equals(one.getName())
                && abba.getAsJsonObject("sections").get(S2).getAsString().equals(two.getName())
                && abba.get("thickness_um").getAsInt() == 40 && abba.get("interval_um").getAsInt() == 400, "The abba block: " + abba);
        require(JsonParser.parseString(Files.readString(output.resolve("snapshots").resolve(AbbaHostSession.SECTIONS_FILE))).equals(abba.get("sections")),
                "abba_sections.json is the sections map");
        require(Files.isRegularFile(output.resolve("snapshots").resolve(S1)), "Snapshot exported");
        require(one.getNumberOfRegistrations() == 0 && two.getNumberOfRegistrations() == 1, "Exporting changes nothing in ABBA");
        AbbaHostSession pages = new AbbaHostSession(mp, output.resolve("pages"), KEY);
        pages.export(sections, Arrays.asList(0, 0), Arrays.asList("ramp"), 25.0, 0, 0);
        ImagePlus multi = new ij.io.Opener().openImage(output.resolve("pages").resolve(S1).toString());
        require(multi.getStackSize() == 2, "One page per exported channel");
        host.attach();
        require(!host.syncPending() && !host.hasPending(), "A fresh attach has nothing to remove or refuse");

        // 4. Checkpoint 1, live: angles, position, affine step and warp step on section one.
        double z0 = one.getSlicingAxisPosition();
        int undoBefore = mp.userActionsSize();
        JsonObject first = row(S1); first.addProperty("position_mm", 5.2); first.add("affine_mm", affine(.3, -.1, 1.05)); first.add("warp", warp(.05));
        AbbaHostSession.ApplyReport report = host.applyCheckpoint(rows(first), angles(4, -1));
        require(report.failed.isEmpty() && report.angles, "Checkpoint 1 applied: " + report.summary());
        anglesAre(mp, 4, -1, "host_angles reach ABBA");
        require(Math.abs(mp.toAtlasZ(one.getSlicingAxisPosition()) - 5.2) < 1e-6, "position_mm reaches ABBA through fromAtlasZ");
        require(names(host, one).equals(Arrays.asList(host.affineName, host.warpName)), "Affine step then warp step: " + names(host, one));
        List<Registration<SourceAndConverter<?>[]>> active = host.activeRegistrations(one);
        warpReadsBack(active.get(1), .05, "Checkpoint 1 warp");
        // ABBA reports a registration's transform as its pull-back (fixed to moving): the inverse of affine_mm.
        AffineTransform3D readAffine = ((AffineTransform3D) active.get(0).getTransformAsRealTransform()).inverse();
        require(Math.abs(readAffine.get(0, 3) - .3) < 1e-12 && Math.abs(readAffine.get(0, 0) - 1.05) < 1e-12, "Affine step reads back: " + readAffine);

        // 5. A new placement with warp:null removes the warp BEFORE replacing the affine.
        JsonObject second = row(S1); second.add("affine_mm", affine(-.2, .15, .95)); second.add("warp", JsonNull.INSTANCE);
        report = host.applyCheckpoint(rows(second), null);
        require(report.failed.isEmpty(), "Checkpoint 2 applied: " + report.summary());
        require(names(host, one).equals(Collections.singletonList(host.affineName)) && one.getNumberOfRegistrations() == 1,
                "Warp removed, affine replaced: " + names(host, one));
        require(Math.abs(((AffineTransform3D) host.activeRegistrations(one).get(0).getTransformAsRealTransform()).inverse().get(0, 3) + .2) < 1e-12, "New affine in place");

        // 6. A warp alone goes on top of the kept affine.
        JsonObject third = row(S1); third.add("warp", warp(.08));
        report = host.applyCheckpoint(rows(third), null);
        require(report.failed.isEmpty() && names(host, one).equals(Arrays.asList(host.affineName, host.warpName)),
                "Warp added on the kept affine: " + names(host, one));
        warpReadsBack(host.activeRegistrations(one).get(1), .08, "Checkpoint 3 warp");

        // 7. One ABBA Undo reverts exactly one checkpoint (here: the warp), Redo restores it.
        mp.cancelLastAction(); mp.waitForTasks();
        require(names(host, one).equals(Collections.singletonList(host.affineName)), "Undo reverts checkpoint 3 only: " + names(host, one));
        // While ABBA's state differs from LangSlice's, a change to that slice is refused, reported and kept.
        JsonObject fourth = row(S1); fourth.add("warp", warp(.02));
        report = host.applyCheckpoint(rows(fourth), null);
        require(report.failed.containsKey(S1) && report.failed.get(S1).contains("changed in ABBA") && host.hasPending(),
                "Outside change is refused and kept: " + report.summary());
        mp.redoAction(); mp.waitForTasks();
        require(names(host, one).equals(Arrays.asList(host.affineName, host.warpName)), "Redo restores checkpoint 3");
        report = host.applyCheckpoint(null, null);
        require(report.failed.isEmpty() && !host.hasPending(), "The kept row is retried and applied: " + report.summary());
        warpReadsBack(host.activeRegistrations(one).get(1), .02, "Retried warp");

        // 8. Cutting angles alone are one undo step too.
        report = host.applyCheckpoint(new JsonArray(), angles(1.5, .5));
        anglesAre(mp, 1.5, .5, "Second host_angles");
        mp.cancelLastAction(); mp.waitForTasks();
        anglesAre(mp, 4, -1, "Undo restores the previous cutting angles");
        mp.redoAction(); mp.waitForTasks();
        anglesAre(mp, 1.5, .5, "Redo restores the new cutting angles");

        // 9. The checkpoints are on ABBA's undo stack, between batch marks.
        require(mp.userActionsSize() > undoBefore && Math.abs(z0 - one.getSlicingAxisPosition()) > 1e-6, "Checkpoints are on ABBA's undo stack");

        // 10. A registration ABBA refuses is reported with ABBA's own reason (its failure list, ABBA 0.25) and leaves the slice as it was.
        int mark = host.failureMark();
        ch.epfl.biop.registration.source.affine.AffineRegistration refusing = new ch.epfl.biop.registration.source.affine.AffineRegistration() {
            @Override public boolean register() { return false; }
            @Override public String getExceptionMessage() { return "probe refused"; }
        };
        refusing.setScijavaContext(mp.getContext());
        int registrationsBefore = two.getNumberOfRegistrations();
        boolean refused = false;
        try { host.appendRegistration(two, refusing); } catch (IllegalStateException expected) { refused = true; }
        mp.waitForTasks();
        require(refused && two.getNumberOfRegistrations() == registrationsBefore, "A refused registration fails and changes nothing");
        String abbaReason = host.abbaFailures(mark, two);
        System.out.println("ABBA_FAILURE mark=" + mark + " reason=" + abbaReason);
        if (mark >= 0) require(abbaReason.contains("probe refused") && host.abbaFailures(mark, one).isEmpty(),
                "ABBA's own failure reason names the refused slice only: " + abbaReason);

        // 11. Native save.
        require(mp.saveState(output.resolve("smoke.abba").toFile(), true), "Native ABBA state saves");
        JsonObject expected = new JsonObject();
        expected.addProperty("slices", 2);
        expected.addProperty("position_mm", mp.toAtlasZ(one.getSlicingAxisPosition()));
        expected.addProperty("rotate_x", mp.getReslicedAtlas().getRotateX());
        expected.addProperty("rotate_y", mp.getReslicedAtlas().getRotateY());
        expected.addProperty("amplitude", .02);
        expected.addProperty("ap_probe", probed); expected.addProperty("ap_converted", converted);
        expected.addProperty("affine_name", host.affineName); expected.addProperty("warp_name", host.warpName);

        // 12. Attach = resync. Another job's step on section two stays: it is the section's own.
        AbbaHostSession other = new AbbaHostSession(mp, output.resolve("other"), OTHER);
        other.adopt(sections); other.attach();
        JsonObject otherRow = row(S2); otherRow.add("affine_mm", affine(.05, .05, 1.0));
        report = other.applyCheckpoint(rows(otherRow), null);
        require(report.failed.isEmpty() && names(host, two).equals(Arrays.asList("Big Warp", other.affineName)), "Another job's step: " + names(host, two));
        // A new attach of this job (another JVM, the menu entry again) finds its two steps on section one and removes them
        // in the same undo step as the job's full state is applied again.
        AbbaHostSession again = new AbbaHostSession(mp, output.resolve("snapshots"), KEY);
        again.load();
        require(again.slices.get(S1) == one && again.slices.get(S2) == two, "load() finds the sections by ABBA slice name");
        again.attach();
        require(again.syncPending() && names(host, one).equals(Arrays.asList(host.affineName, host.warpName)), "attach() changes nothing by itself");
        JsonObject full = row(S1); full.addProperty("position_mm", 5.2); full.add("affine_mm", affine(.1, .1, 1.0)); full.add("warp", warp(.06));
        report = again.applyCheckpoint(rows(full), null);
        require(report.failed.isEmpty() && report.synced.equals(Collections.singletonList(S1)) && report.applied.equals(Collections.singletonList(S1)),
                "The first checkpoint removes this job's steps and applies the job: " + report.toJson() + " synced " + report.synced);
        require(names(host, one).equals(Arrays.asList(host.affineName, host.warpName)) && one.getNumberOfRegistrations() == 2,
                "Exactly this job's steps were replaced: " + names(host, one));
        warpReadsBack(host.activeRegistrations(one).get(1), .06, "Resynced warp");
        require(names(host, two).equals(Arrays.asList("Big Warp", other.affineName)), "Another key's step and the user's own stay: " + names(host, two));
        mp.cancelLastAction(); mp.waitForTasks();
        require(names(host, one).equals(Arrays.asList(host.affineName, host.warpName)), "One Undo reverts the whole sync: " + names(host, one));
        warpReadsBack(host.activeRegistrations(one).get(1), .02, "Undo brings the earlier steps back");
        mp.redoAction(); mp.waitForTasks();
        warpReadsBack(host.activeRegistrations(one).get(1), .06, "Redo brings the sync back");
        // This job's step under a step of someone else's: the section is refused and nothing is removed.
        JsonObject userTransform = new JsonObject(); userTransform.add("affine_mm", affine(.02, 0, 1.0));
        Registration<SourceAndConverter<?>[]> userAffine = foreign.prepareRegistration(userTransform);
        userAffine.setRegistrationName("User affine");
        foreign.appendRegistration(one, userAffine);
        mp.waitForTasks();
        AbbaHostSession buried = new AbbaHostSession(mp, output.resolve("snapshots"), KEY);
        buried.load(); buried.attach();
        require(!buried.syncPending() && buried.pendingReasons().containsKey(S1) && buried.pendingReasons().get(S1).contains("changed in ABBA"),
                "A buried step of this job: the section is refused: " + buried.pendingReasons());
        JsonObject moved = row(S1); moved.addProperty("position_mm", 5.3);
        report = buried.applyCheckpoint(rows(moved), null);
        require(report.failed.containsKey(S1) && report.failed.get(S1).contains("changed in ABBA")
                && names(host, one).equals(Arrays.asList(host.affineName, host.warpName, "User affine"))
                && Math.abs(mp.toAtlasZ(one.getSlicingAxisPosition()) - 5.2) < 1e-6, "Its rows are refused and kept; nothing changed: " + names(host, one));
        require(names(host, two).equals(Arrays.asList("Big Warp", other.affineName)), "The other section is untouched");
        System.out.println("HOST_SESSION_PASS toAtlasZ=probe (" + converted + " vs " + probed + "), tilted export, existing_warp, live position/affine/warp/angles,"
                + " warp-before-affine replace, one-checkpoint undo/redo, outside-change refusal + retry, ABBA failure reason, native save,"
                + " attach: this job's steps resynced in one undo step, another key's and the user's kept, a buried step refused");
        return expected;
    }

    /** Reopens the saved project in a fresh JVM whose classpath has no LangSlice connector classes. */
    static void reload(Path cache, Path output) throws Exception {
        String classpath = System.getProperty("java.class.path");
        StringBuilder without = new StringBuilder();
        for (String entry : classpath.split(java.io.File.pathSeparator)) {
            Path path = Paths.get(entry).toAbsolutePath().normalize();
            // Keep only the Maven dependencies and the test classes' ReloadCheck: never target/classes or the JAR.
            if (path.endsWith(Paths.get("target", "classes")) || path.getFileName().toString().startsWith("langslice-fiji")) continue;
            without.append(without.length() == 0 ? "" : java.io.File.pathSeparator).append(entry);
        }
        Path java = Paths.get(System.getProperty("java.home"), "bin", "java");
        Process process = new ProcessBuilder(java.toString(), "-Djava.awt.headless=true", "--add-opens=java.base/java.lang=ALL-UNNAMED",
                "-cp", without.toString(),
                "org.langslice.fiji.reload.ReloadCheck", cache.toString(), output.toString()).inheritIO().start();
        require(process.waitFor() == 0, "Reload without the connector failed");
    }

    /** An independent AP measurement: ABBA's atlas coordinate images read at the slice's plane (flat session only). */
    static double probeAP(MultiSlicePositioner mp, SliceSources slice) {
        final double[] measured = {Double.NaN};
        SimpleRegistrationPlugin probe = new SimpleRegistrationPlugin() {
            public double getVoxelSizeInMicron() { return 40; }
            public void setRegistrationParameters(Map<String, String> parameters) { }
            public InvertibleRealTransform register(ImagePlus fixed, ImagePlus moving, ImagePlus fm, ImagePlus mm) {
                float center = fixed.getStack().getProcessor(1).getf(fixed.getWidth() / 2, fixed.getHeight() / 2);
                float left = fixed.getStack().getProcessor(1).getf(fixed.getWidth() / 3, fixed.getHeight() / 2);
                float above = fixed.getStack().getProcessor(1).getf(fixed.getWidth() / 2, fixed.getHeight() / 3);
                if (Float.isFinite(center) && Math.abs(center - left) < 1e-4 && Math.abs(center - above) < 1e-4) measured[0] = center;
                return new AffineTransform3D();
            }
        };
        SimpleRegistrationWrapper wrapper = new SimpleRegistrationWrapper("LangSlice axis probe", probe);
        wrapper.setScijavaContext(mp.getContext());
        double[] roi = mp.getROI();
        Map<String, Object> parameters = new HashMap<>();
        parameters.put("px", roi[0]); parameters.put("py", roi[1]); parameters.put("sx", roi[2]); parameters.put("sy", roi[3]); parameters.put("pz", 0);
        wrapper.setRegistrationParameters(MultiSlicePositioner.convertToString(mp.getContext(), parameters));
        wrapper.setTimePoint(0);
        wrapper.setFixedImage(SourcesProcessorHelper.compose(new SourcesZOffset(slice), new SourcesChannelsSelect(Arrays.asList(3, 4, 5)))
                .apply(mp.getReslicedAtlas().nonExtendedSlicedSources));
        wrapper.setMovingImage(SourcesProcessorHelper.compose(new SourcesZOffset(slice), new SourcesChannelsSelect(0)).apply(slice.getRegisteredSources()));
        require(wrapper.register() && Double.isFinite(measured[0]), "Coordinate probe measured the AP plane");
        return measured[0];
    }
}
