package org.langslice.fiji;

import ch.epfl.biop.atlas.aligner.MultiSlicePositioner;
import ch.epfl.biop.atlas.aligner.SliceSources;
import com.google.gson.*;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.*;
import java.time.Duration;
import java.util.*;
import java.util.concurrent.CancellationException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import javax.imageio.ImageIO;
import javax.swing.*;

/** Opens the Registration dialog and runs the worker, leaving the user's ABBA untouched until the end. */
public final class AgentRunner {
    private static final Set<MultiSlicePositioner> RUNNING = Collections.newSetFromMap(new WeakHashMap<>());
    private AgentRunner() { }

    /** Where run messages go: the log window, the compact status window, or a test. */
    interface Progress {
        void line(String text);
        void fragment(String text);
        void status(String text);
    }

    public static void show(MultiSlicePositioner mp, Path environment) {
        if (!SwingUtilities.isEventDispatchThread()) {
            SwingUtilities.invokeLater(() -> show(mp, environment)); return;
        }
        if (RUNNING.contains(mp)) {
            JOptionPane.showMessageDialog(null, "A LangSlice run is already active in this session."); return;
        }
        List<SliceSources> selected = selection(mp);
        if (selected.isEmpty()) { JOptionPane.showMessageDialog(null, "Import sections into ABBA before running LangSlice."); return; }
        boolean chosen = !mp.getSelectedSlices().isEmpty();
        new SwingWorker<JsonObject, Void>() {
            protected JsonObject doInBackground() throws Exception {
                try (WorkerClient worker = new WorkerClient(environment)) {
                    return worker.request("setup.status", new JsonObject(), null, Duration.ofSeconds(60));
                }
            }
            protected void done() {
                try { open(mp, selected, chosen, get()); }
                catch (Exception error) { JOptionPane.showMessageDialog(null, "Could not check LangSlice. Open Setup to reconnect the environment."); SetupDialog.show(mp); }
            }
        }.execute();
    }

    /** The selected slices, or every slice when none is selected, in atlas order. */
    static List<SliceSources> selection(MultiSlicePositioner mp) {
        List<SliceSources> selected = new ArrayList<>(mp.getSelectedSlices());
        if (selected.isEmpty()) selected.addAll(mp.getSlices());
        selected.sort(Comparator.comparingDouble(SliceSources::getSlicingAxisPosition));
        return selected;
    }

    private static void open(MultiSlicePositioner mp, List<SliceSources> selected, boolean chosen, JsonObject status) {
        RegistrationSettings settings = RegistrationSettings.load(RegistrationSettings.PREFS);
        int[] protocol = protocolDefaults(mp);
        if (protocol[0] > 0) settings.interval = protocol[0];
        if (protocol[1] > 0) settings.thickness = protocol[1];
        List<RegistrationDialog.SliceRow> rows = new ArrayList<>();
        int channels = Integer.MAX_VALUE;
        for (SliceSources slice : selected) {
            rows.add(new RegistrationDialog.SliceRow(slice.getName(), slice.getNumberOfRegistrations()));
            channels = Math.min(channels, slice.getRegisteredSources().length);
        }
        List<String> names = new ArrayList<>();
        for (int c = 0; c < channels; c++) {
            String name = selected.get(0).getRegisteredSources()[c].getSpimSource().getName();
            names.add(name == null ? "" : name);
        }
        String caption = chosen ? selected.size() + " slice" + (selected.size() == 1 ? "" : "s") + " selected in ABBA."
                : "No slices selected in ABBA, so all " + selected.size() + " slices are used.";
        new RegistrationDialog(status, rows, caption, names, settings, new AbbaHost(mp, selected, names.size())).setVisible(true);
    }

    /** Section interval and thickness in µm from ABBA's slices; 0 when ABBA gives no value. */
    static int[] protocolDefaults(MultiSlicePositioner mp) {
        List<Double> positions = new ArrayList<>(), thicknesses = new ArrayList<>();
        mp.getSlices().forEach(slice -> {
            double p = slice.getSlicingAxisPosition(), t = slice.getThicknessInMm();
            if (Double.isFinite(p)) positions.add(p);
            if (Double.isFinite(t) && t > 0) thicknesses.add(t);
        });
        Collections.sort(positions); List<Double> gaps = new ArrayList<>();
        for (int i = 1; i < positions.size(); i++) if (positions.get(i) - positions.get(i - 1) > 1e-6) gaps.add(positions.get(i) - positions.get(i - 1));
        return new int[]{medianMicrons(gaps, 0), medianMicrons(thicknesses, 0)};
    }

    static int medianMicrons(List<Double> values, int fallback) {
        if (values.isEmpty()) return fallback;
        Collections.sort(values); int middle = values.size() / 2;
        double median = values.size() % 2 == 0 ? (values.get(middle - 1) + values.get(middle)) / 2 : values.get(middle);
        return Math.max(1, (int) Math.round(median * 1000));
    }

    /** The dialog's view of a live ABBA session. */
    private static final class AbbaHost implements RegistrationDialog.Host {
        final MultiSlicePositioner mp; final List<SliceSources> slices; final int channels;
        final Map<String, BufferedImage> exported = new HashMap<>();
        Path previews;
        AbbaHost(MultiSlicePositioner mp, List<SliceSources> slices, int channels) { this.mp = mp; this.slices = slices; this.channels = channels; }

        public BufferedImage[] preview(int index, List<Integer> pages, JsonObject preprocessing, double pixelSize) throws Exception {
            synchronized (this) { if (previews == null) previews = Files.createTempDirectory("langslice-preview-"); }
            String key = index + "_" + pages.toString().replaceAll("[^0-9]+", "-") + "_" + pixelSize;
            Path snapshot = previews.resolve("slice_" + key + ".tif");
            BufferedImage raw;
            synchronized (exported) {
                raw = exported.get(key);
                if (raw == null) { raw = new AbbaHostSession(mp, previews).exportPreview(slices.get(index), pages, pixelSize, snapshot); exported.put(key, raw); }
            }
            try (WorkerClient worker = new WorkerClient(environment())) {
                return new BufferedImage[]{raw, requestPreview(worker, snapshot, preprocessing, previews.resolve("after_" + System.nanoTime() + ".png"))};
            }
        }
        public JsonObject estimate(JsonObject params) throws Exception {
            try (WorkerClient worker = new WorkerClient(environment())) { return worker.request("linear.estimate", params, null, Duration.ofSeconds(60)); }
        }
        public JsonObject status() throws Exception {
            try (WorkerClient worker = new WorkerClient(environment())) { return worker.request("setup.status", new JsonObject(), null, Duration.ofSeconds(60)); }
        }
        public void setup() { SetupDialog.show(mp); }
        public void run(RegistrationSettings settings, Map<Integer, String> damaged) {
            Map<SliceSources, String> marked = new HashMap<>();
            damaged.forEach((row, note) -> marked.put(slices.get(row), note));
            start(mp, settings, slices, marked, channels);
        }
        public void close() {
            Path folder = previews;
            if (folder == null) return;
            try (java.util.stream.Stream<Path> files = Files.walk(folder)) {
                files.sorted(Comparator.reverseOrder()).forEach(path -> { try { Files.deleteIfExists(path); } catch (IOException ignored) { } });
            } catch (IOException ignored) { }
        }
    }

    private static Path environment() {
        Path environment = EnvironmentDiscovery.saved();
        if (environment == null) throw new IllegalStateException("Open Setup and choose the LangSlice environment.");
        return environment;
    }

    /** The worker writes the exact grayscale image the agent would see for this snapshot. */
    static BufferedImage requestPreview(WorkerClient worker, Path snapshot, JsonObject preprocessing, Path output) throws Exception {
        JsonObject params = new JsonObject();
        params.addProperty("image_path", snapshot.toString());
        params.add("preprocessing", preprocessing);
        params.addProperty("output_path", output.toString());
        JsonObject result = worker.request("preprocess.preview", params, null, Duration.ofSeconds(120));
        Path written = Paths.get(result.has("output_path") ? result.get("output_path").getAsString() : output.toString());
        try {
            BufferedImage image = ImageIO.read(written.toFile());
            if (image == null) throw new IOException("LangSlice did not write a readable preview.");
            return image;
        } finally { Files.deleteIfExists(written); }
    }

    /** Runs linear.run to its result without touching ABBA; remembers each checkpoint's updates_since_start. */
    static JsonObject runLinear(WorkerClient connection, JsonObject request, AtomicReference<JsonArray> partial,
            Progress progress, AtomicBoolean stopping) throws Exception {
        int[] checkpoints = {0};
        return connection.request("linear.run", request, event -> {
            if (stopping.get()) throw new CancellationException();
            if (event.has("payload")) {
                JsonObject payload = event.getAsJsonObject("payload");
                String kind = payload.has("kind") ? payload.get("kind").getAsString() : "";
                if (kind.equals("checkpoint")) {
                    if (payload.has("updates_since_start") && payload.get("updates_since_start").isJsonArray())
                        partial.set(payload.getAsJsonArray("updates_since_start"));
                    boolean initial = payload.has("initial") && payload.get("initial").getAsBoolean();
                    if (!initial) progress.status("The agent is working. Saved steps: " + (++checkpoints[0]) + ". ABBA changes when the run ends.");
                } else if (kind.equals("log") && payload.has("message")) progress.line(payload.get("message").getAsString());
                else if (kind.equals("agent_event")) { String text = agentText(payload.getAsJsonObject("event")); if (!text.isEmpty()) progress.fragment(text); }
            } else if (event.has("message")) progress.line(event.get("message").getAsString());
        }, Duration.ofHours(12));
    }

    /** Initial-to-final updates; older workers without final_updates fall back to the last checkpoint. */
    static JsonArray finalUpdates(JsonObject result, JsonArray lastCheckpoint) {
        if (result.has("final_updates") && result.get("final_updates").isJsonArray()) return result.getAsJsonArray("final_updates");
        if (lastCheckpoint != null) return lastCheckpoint;
        throw new IllegalStateException("This LangSlice version does not report final results. Update LangSlice and the Fiji plugin together.");
    }

    private static void start(MultiSlicePositioner mp, RegistrationSettings settings, List<SliceSources> selected,
            Map<SliceSources, String> damaged, int channels) {
        if (RUNNING.contains(mp)) { JOptionPane.showMessageDialog(null, "A LangSlice run is already active in this session."); return; }
        final Path environment;
        try { environment = environment(); } catch (IllegalStateException missing) { JOptionPane.showMessageDialog(null, missing.getMessage()); return; }
        RUNNING.add(mp);
        RunWindow window = new RunWindow(settings.showLog);
        window.open();
        AtomicReference<WorkerClient> client = new AtomicReference<>();
        AtomicReference<Thread> runningThread = new AtomicReference<>();
        AtomicReference<JsonArray> partial = new AtomicReference<>();
        AtomicReference<AbbaHostSession> session = new AtomicReference<>();
        AtomicBoolean stopping = new AtomicBoolean(), applying = new AtomicBoolean();
        SwingWorker<String, Void> worker = new SwingWorker<String, Void>() {
            protected String doInBackground() throws Exception {
                runningThread.set(Thread.currentThread());
                if (stopping.get()) throw new CancellationException();
                Path folder = Files.createTempDirectory("langslice-abba-");
                window.status("Preparing calibrated snapshots…");
                window.line("Preparing calibrated snapshots in " + folder);
                window.line("Your ABBA session is not changed until the run ends. Avoid editing the listed slices meanwhile.");
                AbbaHostSession host = new AbbaHostSession(mp, folder);
                JsonObject request = host.prepare(settings.spec(), selected, settings.exportChannels(channels),
                        settings.pixelSize, damaged, !settings.overwrite);
                request.add("preprocessing", settings.preprocessing(channels));
                if (settings.saveTraces) request.addProperty("trace_dir", settings.traceDir);
                session.set(host);
                AbbaHostSession.checkInterrupted();
                window.status("The agent is working. ABBA changes when the run ends.");
                JsonObject result;
                try (WorkerClient connection = new WorkerClient(environment)) {
                    client.set(connection);
                    result = runLinear(connection, request, partial, window, stopping);
                } finally { client.set(null); }
                JsonArray updates = finalUpdates(result, partial.get());
                boolean submitted = result.has("state") && result.getAsJsonObject("state").has("submitted")
                        && result.getAsJsonObject("state").get("submitted").getAsBoolean();
                applying.set(true); window.applying();
                window.status("Applying the result to ABBA…");
                host.apply(updates);
                String outputs = result.has("output_dir") ? " Run files: " + result.get("output_dir").getAsString() : "";
                if (result.has("trace_files") && result.getAsJsonArray("trace_files").size() > 0) {
                    StringBuilder saved = new StringBuilder(" Trace saved: ");
                    for (int i = 0; i < result.getAsJsonArray("trace_files").size(); i++)
                        saved.append(i == 0 ? "" : ", ").append(result.getAsJsonArray("trace_files").get(i).getAsString());
                    outputs += saved;
                }
                return (submitted ? "Finished. " : "The agent stopped before submitting; review the result. ")
                        + (updates.size() == 0 ? "The agent made no changes." : "The result was applied to ABBA as one step; ABBA's Undo reverts it.")
                        + " Save the project with ABBA's normal Save command." + outputs;
            }
            protected void done() {
                RUNNING.remove(mp);
                try { window.finish(get()); }
                catch (Exception e) {
                    Throwable cause = e.getCause() == null ? e : e.getCause();
                    boolean stopped = stopping.get() || e instanceof CancellationException || cause instanceof CancellationException;
                    JsonArray last = partial.get();
                    AbbaHostSession host = session.get();
                    boolean offer = !applying.get() && host != null && last != null && last.size() > 0;
                    String message = applying.get()
                            ? "Applying the result to ABBA failed: " + cause.getMessage() + " Use ABBA's Undo to revert any partial change."
                            : (stopped ? "Stopped. Your ABBA session was not changed." : "Run stopped: " + cause.getMessage() + " Your ABBA session was not changed.");
                    if (offer) message += " You can apply the last saved step of the run to ABBA as one undoable step.";
                    if (settings.saveTraces) message += " The trace so far is in " + settings.traceDir + ".";
                    window.finish(message);
                    if (offer) window.offerPartial(() -> applyPartial(window, host, last));
                }
            }
        };
        window.stop.addActionListener(event -> {
            stopping.set(true); window.stop.setEnabled(false);
            WorkerClient connection = client.get(); if (connection != null) connection.close();
            Thread thread = runningThread.get(); if (connection == null && thread != null && !applying.get()) thread.interrupt();
        });
        worker.execute();
    }

    private static void applyPartial(RunWindow window, AbbaHostSession host, JsonArray updates) {
        window.status("Applying the partial result to ABBA…");
        new SwingWorker<Void, Void>() {
            protected Void doInBackground() { host.apply(updates); return null; }
            protected void done() {
                try { get(); window.partialDone("The partial result was applied to ABBA as one step; ABBA's Undo reverts it. Save the project with ABBA's normal Save command."); }
                catch (Exception e) {
                    Throwable cause = e.getCause() == null ? e : e.getCause();
                    window.partialDone("Applying the partial result failed: " + cause.getMessage() + " Use ABBA's Undo to revert any partial change.");
                }
            }
        }.execute();
    }

    static String agentText(JsonObject event) {
        String kind = event.has("kind") ? event.get("kind").getAsString() : "";
        if ((kind.equals("text") || kind.equals("reasoning")) && event.has("text")) return event.get("text").getAsString();
        if (kind.equals("tool_start")) return "\nWorking: " + (event.has("name") ? event.get("name").getAsString().replace('_', ' ') : "registration") + "\n";
        if (kind.equals("tool_result") && event.has("response") && event.get("response").isJsonObject()) {
            JsonObject response = event.getAsJsonObject("response");
            String status = response.has("status") ? response.get("status").getAsString() : "";
            if (!status.isEmpty() && !status.equals("ok") && !status.equals("success")) {
                String reason = response.has("message") ? response.get("message").getAsString() : response.has("reason") ? response.get("reason").getAsString() : status;
                return "\n" + reason.substring(0, Math.min(500, reason.length())) + "\n";
            }
        }
        if (kind.equals("error") && event.has("text")) return "\nRun failed: " + event.get("text").getAsString() + "\n";
        return "";
    }
}
