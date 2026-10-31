package org.langslice.fiji;

import bdv.viewer.SourceAndConverter;
import ch.epfl.biop.atlas.aligner.MultiSlicePositioner;
import ch.epfl.biop.atlas.aligner.SliceSources;
import ch.epfl.biop.registration.Registration;
import com.google.gson.*;
import ij.ImagePlus;
import net.imglib2.realtransform.RealTransform;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.time.Duration;
import java.util.*;
import java.util.concurrent.atomic.AtomicReference;

/**
 * End to end, headless: a real ABBA 0.25.0, the REAL worker (protocol 3) and the agent CLI. The connector prepares the
 * job (snapshots, abba.prepare) and starts the follower; {@code connectors/fiji/e2e/driver.py} stands in for the user's
 * agent app, calling {@code langslice-job FOLDER VERB} as processes; ABBA must end with the job's positions, angles and
 * steps. Then the session is closed (the follower ends), the saved project reopened in a new session and attached again:
 * one sync step, identical geometry. No model and no window. Prints E2E_PASS.
 *
 * <p>Args: ATLAS_CACHE ENV_PREFIX DRIVER_PY OUTPUT_DIR (ENV_PREFIX: an environment whose {@code bin/python} runs this
 * checkout's LangSlice).
 */
public final class E2EFollowCheck {
    static { net.imagej.patcher.LegacyInjector.preinit(); }

    static void require(boolean condition, String message) { if (!condition) throw new AssertionError(message); }

    public static void main(String[] args) throws Exception {
        Path cache = Paths.get(args[0]), env = Paths.get(args[1]), driver = Paths.get(args[2]), output = Paths.get(args[3]);
        Files.createDirectories(output);
        ProjectJob.useRoot(output.resolve("projects"));

        // 1. The real worker speaks protocol 3.
        JsonObject status;
        try (WorkerClient worker = new WorkerClient(env)) { status = worker.request("setup.status", new JsonObject(), null, Duration.ofSeconds(120)); }
        System.out.println("E2E setup.status: " + status);
        require(status.get("protocol_version").getAsInt() == OpenLangSliceCommand.PROTOCOL_VERSION, "protocol_version 3: " + status.get("protocol_version"));

        // 2. A real ABBA with two imported sections.
        net.imagej.ImageJ imagej = new net.imagej.ImageJ();
        MultiSlicePositioner mp = HostSessionSmokeTest.start(imagej, cache);
        for (int i = 0; i < 2; i++) {
            ImagePlus image = ij.IJ.createImage("Synthetic section " + i, "8-bit ramp", 160, 120, 1);
            image.getCalibration().pixelWidth = 25; image.getCalibration().pixelHeight = 25; image.getCalibration().setUnit("um");
            new ij.io.FileSaver(image).saveAsTiff(output.resolve("synthetic" + i + ".tif").toString()); image.close();
        }
        imagej.command().run(ch.epfl.biop.atlas.aligner.command.ImportSlicesFromFilesCommand.class, true,
                "mp", mp, "datasetname", "synthetic",
                "files", new java.io.File[]{output.resolve("synthetic0.tif").toFile(), output.resolve("synthetic1.tif").toFile()},
                "split_rgb_channels", false, "first_slice_position_mm", 5.0, "slice_spacing_mm", .4).get();
        mp.waitForTasks();
        List<SliceSources> sections = OpenLangSliceCommand.selection(mp);
        require(sections.size() == 2, "two sections");

        // 3. The menu entry's flow without its windows: a new project folder, the prepared job, the follower.
        ProjectJob.Found found = ProjectJob.find(mp);
        require(found.folder == null && found.candidates.isEmpty(), "a new project");
        Path folder = ProjectJob.root().resolve(found.key);
        OpenLangSliceCommand.prepareJob(mp, env, folder, sections, false);
        Path jobFolder = ProjectJob.jobFolder(folder);
        require(ProjectJob.hasJob(folder) && !ProjectJob.hasCheckpoint(folder), "abba.prepare wrote job.json and no checkpoint");
        require(Files.isRegularFile(folder.resolve(ProjectJob.PROJECT_FILE)) && Files.isRegularFile(folder.resolve("section_0001.tif")), "project file and snapshots");
        System.out.println("E2E job.json: " + Files.readString(jobFolder.resolve("job.json")).replaceAll("\\s+", " "));
        require(folder.equals(ProjectJob.find(mp).folder), "the session finds its folder again");
        int undoBefore = mp.userActionsSize();
        Follower follower = Follower.ensure(mp, env, folder);
        require(follower == Follower.following(mp) && follower.alive(), "the follower runs");
        require(follower == Follower.ensure(mp, env, folder), "the menu entry again keeps a follower with nothing refused");

        // 4. The driver stands in for the agent app; every write reaches ABBA through the follower.
        Process python = new ProcessBuilder(EnvironmentDiscovery.python(env).toString(), "-u", driver.toString(), folder.toString())
                .redirectErrorStream(true).start();
        AtomicReference<JsonObject> finalState = new AtomicReference<>();
        try (BufferedReader lines = new BufferedReader(new InputStreamReader(python.getInputStream(), StandardCharsets.UTF_8))) {
            String line;
            while ((line = lines.readLine()) != null) {
                System.out.println(line);
                if (line.startsWith("DRIVER final: ")) finalState.set(JsonParser.parseString(line.substring("DRIVER final: ".length())).getAsJsonObject());
            }
        }
        require(python.waitFor() == 0 && finalState.get() != null, "the driver ended cleanly");
        // The warps take the worker tens of seconds to convert: wait until it says ABBA has the saved state.
        Path stateFile = jobFolder.resolve("state.json");
        long driverEnded = System.nanoTime();
        require(awaitCaughtUp(follower, stateFile, Duration.ofSeconds(300)), "the follower caught up with the job");
        System.out.printf("E2E caught up %.1f s after the driver ended%n", (System.nanoTime() - driverEnded) / 1e9);
        JsonObject expected = finalState.get();
        require(expected.get("submitted").getAsBoolean(), "the job was submitted");
        AbbaHostSession reader = new AbbaHostSession(mp, folder, found.key);
        reader.load();
        String why = awaitJob(mp, reader, expected, Duration.ofSeconds(90));
        require(why == null, "ABBA did not reach the job's state: " + why);
        Path applied = jobFolder.resolve(Follower.APPLIED_LOG);
        List<String> appliedLines = awaitStable(applied);
        System.out.println("E2E applied checkpoints: " + appliedLines.size() + "; ABBA undo stack grew by " + (mp.userActionsSize() - undoBefore));
        appliedLines.forEach(line -> System.out.println("E2E applied: " + line.substring(0, Math.min(300, line.length()))));
        for (String line : appliedLines)
            require(JsonParser.parseString(line).getAsJsonObject().getAsJsonObject("failed").size() == 0, "every row reached ABBA: " + line);
        require(mp.userActionsSize() > undoBefore, "the checkpoints are on ABBA's undo stack");
        Map<String, List<String>> names = names(reader);
        Map<String, double[]> geometry = geometry(mp, reader);
        System.out.println("E2E steps: " + names);

        // 5. Save, close the session (the follower ends), reopen the project in a new session.
        Path saved = output.resolve("e2e.abba");
        require(mp.saveState(saved.toFile(), true), "native save");
        double rotateX = mp.getReslicedAtlas().getRotateX(), rotateY = mp.getReslicedAtlas().getRotateY();
        mp.close();
        require(follower.awaitEnd(Duration.ofSeconds(15)) && !follower.alive() && Follower.following(mp) == null, "closing ABBA's session ends the follower");
        MultiSlicePositioner reopened = HostSessionSmokeTest.start(imagej, cache);
        require(reopened.loadState(saved.toFile()), "the project reopens");
        reopened.waitForTasks();
        require(ProjectJob.key(reopened).equals(found.key) && folder.equals(ProjectJob.find(reopened).folder), "the reopened project has the same key and folder");
        require(Math.abs(reopened.getReslicedAtlas().getRotateX() - rotateX) < 1e-12 && Math.abs(reopened.getReslicedAtlas().getRotateY() - rotateY) < 1e-12,
                "the cutting angles reopen");

        // 6. Attach again: one sync step, identical geometry.
        AbbaHostSession again = new AbbaHostSession(reopened, folder, found.key);
        again.load();
        require(names(again).equals(names) && close(geometry(reopened, again), geometry), "the saved project holds the job's steps");
        int before = appliedLines.size();
        long attached = System.nanoTime();
        Follower resumed = Follower.ensure(reopened, env, folder);
        require(awaitCaughtUp(resumed, stateFile, Duration.ofSeconds(120)), "the attach caught up with the job");
        System.out.printf("E2E resync caught up %.1f s after attaching%n", (System.nanoTime() - attached) / 1e9);
        long deadline = System.nanoTime() + Duration.ofSeconds(60).toNanos();
        while (System.nanoTime() < deadline && lines(applied).size() == before) Thread.sleep(200);
        List<String> after = awaitStable(applied);
        System.out.println("E2E resync: " + after.get(after.size() - 1));
        require(after.size() == before + 1, "attaching again applies one checkpoint: " + (after.size() - before));
        require(names(again).equals(names) && close(geometry(reopened, again), geometry), "identical geometry after the sync: " + names(again));
        reopened.cancelLastAction(); reopened.waitForTasks();
        require(names(again).equals(names) && close(geometry(reopened, again), geometry),
                "one ABBA Undo reverts the whole sync, removals included, back to the saved steps: " + names(again));
        reopened.redoAction(); reopened.waitForTasks();
        require(names(again).equals(names) && close(geometry(reopened, again), geometry), "Redo brings the sync back");
        require(Follower.stop(reopened) && !resumed.alive(), "Stop following ends it");
        System.out.println("E2E_PASS");
        imagej.context().dispose();
        System.exit(0);
    }

    /** True once the follower has applied everything up to the job's saved state (its {@code idle} event). */
    static boolean awaitCaughtUp(Follower follower, Path stateFile, Duration timeout) throws Exception {
        long saved = Files.getLastModifiedTime(stateFile).to(java.util.concurrent.TimeUnit.NANOSECONDS);
        long deadline = System.nanoTime() + timeout.toNanos();
        while (System.nanoTime() < deadline) {
            if (follower.caughtUpTo() == saved) return true;
            Thread.sleep(200);
        }
        return false;
    }

    /** null once ABBA shows the job's positions, angles and steps; else what still differs after *timeout*. */
    static String awaitJob(MultiSlicePositioner mp, AbbaHostSession reader, JsonObject expected, Duration timeout) throws InterruptedException {
        long deadline = System.nanoTime() + timeout.toNanos();
        String why;
        do {
            why = differs(mp, reader, expected);
            if (why == null) return null;
            Thread.sleep(250);
        } while (System.nanoTime() < deadline);
        return why;
    }

    static String differs(MultiSlicePositioner mp, AbbaHostSession reader, JsonObject expected) {
        double pitch = -Math.toDegrees(mp.getReslicedAtlas().getRotateX()), yaw = -Math.toDegrees(mp.getReslicedAtlas().getRotateY());
        if (Math.abs(pitch - expected.get("pitch_deg").getAsDouble()) > 1e-6 || Math.abs(yaw - expected.get("yaw_deg").getAsDouble()) > 1e-6)
            return "angles " + pitch + "/" + yaw + " vs " + expected.get("pitch_deg") + "/" + expected.get("yaw_deg");
        for (Map.Entry<String, JsonElement> entry : expected.getAsJsonObject("sections").entrySet()) {
            JsonObject job = entry.getValue().getAsJsonObject();
            SliceSources slice = reader.slices.get(entry.getKey());
            double position = mp.toAtlasZ(slice.getSlicingAxisPosition());
            if (Math.abs(position - job.get("position_mm").getAsDouble()) > 1e-6) return entry.getKey() + " at " + position + " vs " + job.get("position_mm");
            List<String> want = new ArrayList<>();
            if (job.get("transform").getAsBoolean() || job.get("flip").getAsBoolean() || job.get("rotation_deg").getAsInt() != 0) want.add(reader.affineName);
            if (job.get("deformation").getAsBoolean()) want.add(reader.warpName);
            List<String> have = stepNames(reader, slice);
            if (!have.equals(want)) return entry.getKey() + " steps " + have + " vs " + want;
        }
        return null;
    }

    static List<String> stepNames(AbbaHostSession reader, SliceSources slice) {
        List<String> names = new ArrayList<>();
        for (Registration<SourceAndConverter<?>[]> registration : reader.activeRegistrations(slice)) names.add(registration.getRegistrationName());
        return names;
    }

    static Map<String, List<String>> names(AbbaHostSession reader) {
        Map<String, List<String>> names = new TreeMap<>();
        reader.slices.forEach((id, slice) -> names.put(id, stepNames(reader, slice)));
        return names;
    }

    /** Per section: its AP position, then a 5x5 grid of points through every active step's transform, in order. */
    static Map<String, double[]> geometry(MultiSlicePositioner mp, AbbaHostSession reader) {
        Map<String, double[]> out = new TreeMap<>();
        reader.slices.forEach((id, slice) -> {
            List<Double> values = new ArrayList<>();
            values.add(mp.toAtlasZ(slice.getSlicingAxisPosition()));
            List<Registration<SourceAndConverter<?>[]>> steps = reader.activeRegistrations(slice);
            for (int i = 0; i < 25; i++) {
                double[] point = {(i % 5 - 2) * .5, (i / 5 - 2) * .4, 0}, next = new double[3];
                for (Registration<SourceAndConverter<?>[]> step : steps) {
                    RealTransform transform = step.getTransformAsRealTransform();
                    transform.apply(point, next);
                    point = next.clone();
                }
                values.add(point[0]); values.add(point[1]);
            }
            out.put(id, values.stream().mapToDouble(Double::doubleValue).toArray());
        });
        return out;
    }

    static boolean close(Map<String, double[]> a, Map<String, double[]> b) {
        if (!a.keySet().equals(b.keySet())) return false;
        for (String id : a.keySet())
            for (int i = 0; i < a.get(id).length; i++) if (!(Math.abs(a.get(id)[i] - b.get(id)[i]) < 1e-6)) return false;
        return true;
    }

    static List<String> lines(Path file) {
        try { return Files.readAllLines(file, StandardCharsets.UTF_8); } catch (java.io.IOException absent) { return Collections.emptyList(); }
    }

    /** The file's lines once they stopped changing for two seconds (the follower polls every half second). */
    static List<String> awaitStable(Path file) throws InterruptedException {
        List<String> last = lines(file);
        long stableSince = System.nanoTime();
        while (System.nanoTime() - stableSince < 2_000_000_000L) {
            Thread.sleep(250);
            List<String> now = lines(file);
            if (!now.equals(last)) { last = now; stableSince = System.nanoTime(); }
        }
        return last;
    }
}
