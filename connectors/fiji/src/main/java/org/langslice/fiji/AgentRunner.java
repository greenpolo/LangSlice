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

/** Opens the Registration dialog and runs the worker; every saved step of the agent lands live in ABBA. */
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
                catch (Exception error) { JOptionPane.showMessageDialog(null, "Could not check LangSlice. Open Setup to reconnect the environment."); SetupDialog.open(); }
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
        new RegistrationDialog(status, rows, caption, names, settings, new AbbaHost(mp, selected, names)).setVisible(true);
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
        final MultiSlicePositioner mp; final List<SliceSources> slices; final List<String> channels;
        final Map<String, BufferedImage> exported = new HashMap<>();
        Path previews;
        AbbaHost(MultiSlicePositioner mp, List<SliceSources> slices, List<String> channels) { this.mp = mp; this.slices = slices; this.channels = channels; }

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
        public void setup() { SetupDialog.open(); }
        public void run(RegistrationSettings settings, Map<Integer, String> damaged, Set<Integer> nonlinearSkip) {
            Map<SliceSources, String> marked = new HashMap<>();
            damaged.forEach((row, note) -> marked.put(slices.get(row), note));
            Set<SliceSources> skip = new HashSet<>();
            nonlinearSkip.forEach(row -> skip.add(slices.get(row)));
            List<SliceSources> sent = new ArrayList<>(slices);
            // Nonlinear alone has nothing to do on a slice it skips: leave it out of the run.
            if (!settings.positioning && !settings.linear) { sent.removeAll(skip); skip.clear(); }
            if (sent.isEmpty()) { JOptionPane.showMessageDialog(null, "No slice is left for LangSlice to work on."); return; }
            start(mp, settings, sent, marked, skip, channels);
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
        Path environment = EnvironmentDiscovery.current();
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

    /**
     * Runs linear.run to its result. Each saved checkpoint goes to live (applied to ABBA as it arrives),
     * and every checkpoint and agent event to registered {@link LangSliceEvents} listeners.
     */
    static JsonObject runLinear(WorkerClient connection, JsonObject request, Progress progress,
            AtomicBoolean stopping, java.util.function.Consumer<JsonObject> live) throws Exception {
        int[] checkpoints = {0};
        return connection.request("linear.run", request,
                event -> handleEvent(event, progress, stopping, checkpoints, live), Duration.ofHours(12));
    }

    /** Shared event handling for both modes: every checkpoint after the initial one is applied live. */
    static void handleEvent(JsonObject event, Progress progress, AtomicBoolean stopping, int[] checkpoints,
            java.util.function.Consumer<JsonObject> live) {
        if (stopping.get()) throw new CancellationException();
        if (event.has("payload")) {
            JsonObject payload = event.getAsJsonObject("payload");
            String kind = payload.has("kind") ? payload.get("kind").getAsString() : "";
            if (!kind.isEmpty()) LangSliceEvents.forward(payload);
            if (kind.equals("checkpoint")) {
                boolean initial = payload.has("initial") && payload.get("initial").getAsBoolean();
                if (!initial) {
                    if (live != null) live.accept(payload);
                    progress.status("The agent is working. Saved steps: " + (++checkpoints[0])
                            + ". Changes are live in ABBA; each saved step is one ABBA Undo.");
                }
            } else if (kind.equals("log") && payload.has("message")) progress.line(payload.get("message").getAsString());
            else if (kind.equals("agent_event") && payload.has("event") && payload.get("event").isJsonObject()) {
                String text = agentText(payload.getAsJsonObject("event"));
                if (!text.isEmpty()) progress.fragment(text);
            }
        } else if (event.has("message")) progress.line(event.get("message").getAsString());
    }

    /** One checkpoint into ABBA; rows ABBA could not take are reported and retried with the next checkpoint. */
    static AbbaHostSession.ApplyReport applyLive(AbbaHostSession host, JsonObject payload, Progress progress) {
        JsonArray rows = payload.has("host_updates") && payload.get("host_updates").isJsonArray() ? payload.getAsJsonArray("host_updates") : new JsonArray();
        JsonObject angles = payload.has("host_angles") && payload.get("host_angles").isJsonObject() ? payload.getAsJsonObject("host_angles") : null;
        AbbaHostSession.ApplyReport report;
        try { report = host.applyCheckpoint(rows, angles); }
        catch (RuntimeException failure) {
            // Never end the run over ABBA: the rows stay pending in the session and are retried next time.
            progress.line("This step could not reach ABBA: " + failure.getMessage() + " It is retried with the next step.");
            return null;
        }
        if (!report.failed.isEmpty() || report.anglesFailed)
            progress.line("Not in ABBA yet (retried with the next step): " + report.summary());
        LangSliceEvents.publish("applied", report.toJson());
        return report;
    }

    /** "" when everything reached ABBA; otherwise what is still missing, for the final message. */
    static String pendingText(AbbaHostSession host) {
        if (host == null || !host.hasPending()) return "";
        Map<String, String> reasons = host.pendingReasons();
        StringBuilder text = new StringBuilder(" Some changes are not in ABBA: ");
        if (reasons.isEmpty()) text.append("the cutting angles. ");
        reasons.forEach((id, reason) -> text.append(id).append(" (").append(host.slices.get(id) == null ? "?" : host.slices.get(id).getName())
                .append("): ").append(reason).append(' '));
        return text.append("Use Retry failed updates to try again.").toString();
    }

    private static String outputs(JsonObject result) {
        String outputs = result.has("output_dir") ? " Run files: " + result.get("output_dir").getAsString() : "";
        if (result.has("trace_files") && result.getAsJsonArray("trace_files").size() > 0) {
            StringBuilder saved = new StringBuilder(" Trace saved: ");
            for (int i = 0; i < result.getAsJsonArray("trace_files").size(); i++)
                saved.append(i == 0 ? "" : ", ").append(result.getAsJsonArray("trace_files").get(i).getAsString());
            outputs += saved;
        }
        return outputs;
    }

    /** Prepares snapshots and the request shared by both modes; registers the run with event listeners. */
    private static JsonObject prepareRun(AbbaHostSession host, RegistrationSettings settings,
            List<SliceSources> selected, Map<SliceSources, String> damaged, Set<SliceSources> nonlinearSkip,
            List<String> channelNames, String mode) throws IOException {
        JsonObject request = host.prepare(settings.spec(), selected, settings.exportChannels(channelNames.size()), channelNames,
                settings.pixelSize, damaged, !settings.overwrite);
        request.add("preprocessing", settings.preprocessing(channelNames.size()));
        if (settings.saveTraces) request.addProperty("trace_dir", settings.traceDir);
        JsonArray skip = new JsonArray();
        host.slices.forEach((file, slice) -> { if (nonlinearSkip.contains(slice)) skip.add(file); });
        if (skip.size() > 0) request.add("nonlinear_skip", skip);
        JsonObject started = new JsonObject();
        started.addProperty("mode", mode);
        started.addProperty("viewer", settings.viewer);
        started.addProperty("image_folder", host.folder.toString());
        LangSliceEvents.runStarted(host.slices, started);
        return request;
    }

    private static void ended(String message, String jobDir) {
        JsonObject body = new JsonObject(); body.addProperty("message", message);
        if (jobDir != null) body.addProperty("job_dir", jobDir);
        LangSliceEvents.publish("run_finished", body);
    }

    private static void start(MultiSlicePositioner mp, RegistrationSettings settings, List<SliceSources> selected,
            Map<SliceSources, String> damaged, Set<SliceSources> nonlinearSkip, List<String> channelNames) {
        if (RUNNING.contains(mp)) { JOptionPane.showMessageDialog(null, "A LangSlice run is already active in this session."); return; }
        final Path environment;
        try { environment = environment(); } catch (IllegalStateException missing) { JOptionPane.showMessageDialog(null, missing.getMessage()); return; }
        if (settings.claude) { startClaude(mp, environment, settings, selected, damaged, nonlinearSkip, channelNames); return; }
        RUNNING.add(mp);
        RunWindow window = new RunWindow(settings.showLog);
        window.open();
        AtomicReference<WorkerClient> client = new AtomicReference<>();
        AtomicReference<Thread> runningThread = new AtomicReference<>();
        AtomicReference<AbbaHostSession> session = new AtomicReference<>();
        AtomicReference<String> jobDir = new AtomicReference<>();
        AtomicBoolean stopping = new AtomicBoolean();
        SwingWorker<String, Void> worker = new SwingWorker<String, Void>() {
            protected String doInBackground() throws Exception {
                runningThread.set(Thread.currentThread());
                if (stopping.get()) throw new CancellationException();
                Path folder = Files.createTempDirectory("langslice-abba-");
                window.status("Preparing calibrated snapshots…");
                window.line("Preparing calibrated snapshots in " + folder);
                window.line("The agent's changes appear in ABBA as it saves them; each saved step is one ABBA Undo. "
                        + "Avoid editing the listed slices while the agent works.");
                AbbaHostSession host = new AbbaHostSession(mp, folder);
                JsonObject request = prepareRun(host, settings, selected, damaged, nonlinearSkip, channelNames, RegistrationSettings.PROVIDER);
                session.set(host);
                AbbaHostSession.checkInterrupted();
                window.status("The agent is working. Changes are live in ABBA.");
                JsonObject result;
                try (WorkerClient connection = new WorkerClient(environment)) {
                    client.set(connection);
                    result = runLinear(connection, request, window, stopping, payload -> applyLive(host, payload, window));
                } finally { client.set(null); }
                if (result.has("output_dir")) jobDir.set(result.get("output_dir").getAsString());
                // Every checkpoint, the final one included, has already reached ABBA; retry what did not, once.
                if (host.hasPending()) host.applyCheckpoint(null, null);
                boolean submitted = result.has("state") && result.getAsJsonObject("state").has("submitted")
                        && result.getAsJsonObject("state").get("submitted").getAsBoolean();
                return (submitted ? "Finished. " : "The agent stopped before submitting; review the result. ")
                        + "The agent's changes are in ABBA; each saved step is one ABBA Undo."
                        + pendingText(host) + " Save the project with ABBA's normal Save command." + outputs(result);
            }
            protected void done() {
                RUNNING.remove(mp);
                AbbaHostSession host = session.get();
                String message;
                try { message = get(); }
                catch (Exception e) {
                    Throwable cause = e.getCause() == null ? e : e.getCause();
                    boolean stopped = stopping.get() || e instanceof CancellationException || cause instanceof CancellationException;
                    message = (stopped ? "Stopped." : "Run stopped: " + cause.getMessage())
                            + (host == null ? " Your ABBA session was not changed."
                                : " The changes the agent saved before that are in ABBA; each saved step is one ABBA Undo.")
                            + pendingText(host);
                    if (settings.saveTraces) message += " The trace so far is in " + settings.traceDir + ".";
                }
                window.finish(message);
                ended(message, jobDir.get());
                if (host != null && host.hasPending()) window.offerRetry(() -> retry(window, host));
            }
        };
        window.stop.addActionListener(event -> {
            stopping.set(true); window.stop.setEnabled(false);
            WorkerClient connection = client.get(); if (connection != null) connection.close();
            Thread thread = runningThread.get(); if (connection == null && thread != null) thread.interrupt();
        });
        worker.execute();
    }

    private static void startClaude(MultiSlicePositioner mp, Path environment, RegistrationSettings settings,
            List<SliceSources> selected, Map<SliceSources, String> damaged, Set<SliceSources> nonlinearSkip, List<String> channelNames) {
        RUNNING.add(mp);
        RunWindow window = new RunWindow(settings.showLog);
        AtomicReference<ClaudeHostChannel> listener = new AtomicReference<>();
        AtomicReference<AbbaHostSession> session = new AtomicReference<>();
        AtomicReference<String> jobDir = new AtomicReference<>();
        AtomicBoolean stopping = new AtomicBoolean();
        Runnable stop = () -> {
            stopping.set(true);
            ClaudeHostChannel channel = listener.get(); if (channel != null) channel.close();
        };
        window.stop.setText("Disconnect"); window.close.setEnabled(true);
        window.stop.addActionListener(e -> stop.run());
        window.close.addActionListener(e -> stop.run());
        window.frame.addWindowListener(new java.awt.event.WindowAdapter() {
            @Override public void windowClosing(java.awt.event.WindowEvent event) { stop.run(); }
        });
        window.open();
        new SwingWorker<String, Void>() {
            protected String doInBackground() throws Exception {
                Path root = Paths.get(System.getProperty("user.home"), ".langslice", "snapshots");
                Files.createDirectories(root);
                Path folder = Files.createTempDirectory(root, "claude-");
                window.status("Preparing calibrated snapshots…");
                AbbaHostSession host = new AbbaHostSession(mp, folder);
                JsonObject request = prepareRun(host, settings, selected, damaged, nonlinearSkip, channelNames, "claude");
                session.set(host);
                request.addProperty("notes", "");
                if (stopping.get()) throw new CancellationException();
                try (ClaudeHostChannel channel = new ClaudeHostChannel()) {
                    listener.set(channel);
                    if (stopping.get()) throw new CancellationException();
                    request.add("host_channel", channel.settings());
                    JsonObject prepared;
                    try (WorkerClient worker = new WorkerClient(environment)) {
                        prepared = worker.request("claude.prepare", request, null, Duration.ofMinutes(5));
                    }
                    if (stopping.get()) throw new CancellationException();
                    JsonObject job = new JsonObject();
                    if (prepared.has("job_id")) job.add("job_id", prepared.get("job_id"));
                    job.add("job_dir", prepared.get("job_dir"));
                    jobDir.set(prepared.get("job_dir").getAsString());
                    LangSliceEvents.publish("job", job);
                    String prompt = prepared.get("prompt").getAsString();
                    SwingUtilities.invokeAndWait(() -> java.awt.Toolkit.getDefaultToolkit().getSystemClipboard()
                            .setContents(new java.awt.datatransfer.StringSelection(prompt), null));
                    window.status("Prompt copied. Paste it into Claude Desktop or Claude Code. Keep this window open for live ABBA updates.");
                    window.line("Saved job: " + prepared.get("job_dir").getAsString());
                    window.line("Avoid editing the selected slices while Claude is working. Closing this window disconnects ABBA; Claude can continue saving results.");
                    int[] checkpoints = {0};
                    channel.receive(event -> handleEvent(event, window, stopping, checkpoints, payload -> applyLive(host, payload, window)));
                    // Checkpoints already applied every change, including changes that were later undone.
                    if (host.hasPending()) host.applyCheckpoint(null, null);
                    return "Claude submitted. Changes are live in ABBA; each saved step is one ABBA Undo." + pendingText(host)
                            + " Save using ABBA's Save command. Results: " + prepared.get("job_dir").getAsString();
                } finally { listener.set(null); }
            }
            protected void done() {
                RUNNING.remove(mp);
                AbbaHostSession host = session.get();
                String message;
                if (stopping.get()) message = "Disconnected. Applied changes remain in ABBA; Claude can continue saving results." + pendingText(host);
                else {
                    try { message = get(); }
                    catch (Exception e) {
                        Throwable cause = e.getCause() == null ? e : e.getCause();
                        message = "Claude connection ended: " + cause.getMessage()
                                + " Applied changes remain in ABBA; results remain in the saved job directory." + pendingText(host);
                    }
                }
                if (stopping.get()) window.status(message); else window.finish(message);
                ended(message, jobDir.get());
                if (host != null && host.hasPending()) window.offerRetry(() -> retry(window, host));
            }
        }.execute();
    }

    /** Applies the updates kept after failures, as one ABBA undo step. */
    private static void retry(RunWindow window, AbbaHostSession host) {
        window.status("Applying the kept updates to ABBA…");
        new SwingWorker<AbbaHostSession.ApplyReport, Void>() {
            protected AbbaHostSession.ApplyReport doInBackground() { return host.applyCheckpoint(null, null); }
            protected void done() {
                try {
                    AbbaHostSession.ApplyReport report = get();
                    LangSliceEvents.publish("applied", report.toJson());
                    boolean again = host.hasPending();
                    window.retryDone(again ? "Still not in ABBA: " + report.summary()
                            : "The kept updates are in ABBA as one step; ABBA's Undo reverts it. Save the project with ABBA's normal Save command.", again);
                } catch (Exception e) {
                    Throwable cause = e.getCause() == null ? e : e.getCause();
                    window.retryDone("Applying the kept updates failed: " + cause.getMessage(), host.hasPending());
                }
            }
        }.execute();
    }

    /**
     * The run log's text for one agent event. A tool's failure is read from its tool_end (sent in both modes:
     * the worker's toolbox emits it around every executed tool), so a refusal shows once, with its reason.
     */
    static String agentText(JsonObject event) {
        String kind = event.has("kind") ? event.get("kind").getAsString() : "";
        if ((kind.equals("text") || kind.equals("reasoning")) && event.has("text")) return event.get("text").getAsString();
        if (kind.equals("tool_start")) return "\nWorking: " + (event.has("name") ? event.get("name").getAsString().replace('_', ' ') : "registration") + "\n";
        if (kind.equals("tool_end") && event.has("response") && event.get("response").isJsonObject()) {
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
