package org.langslice.fiji;

import ch.epfl.biop.atlas.aligner.MultiSlicePositioner;
import com.google.gson.*;

import java.io.IOException;
import java.nio.file.Path;
import java.time.Duration;
import java.util.*;

/**
 * Mirrors one job into one ABBA session: the worker's {@code abba.follow} reads the job's {@code state.json} as it
 * changes and streams checkpoints; each is applied as one ABBA undo step. One follower per ABBA session; ABBA's
 * closing ends it (the worker sees stdin end and finishes), and nothing else does except {@link #stop}.
 */
final class Follower {
    /** Applies one checkpoint to ABBA (the session's {@link AbbaHostSession#applyCheckpoint}; a test's recorder). */
    interface Target {
        AbbaHostSession.ApplyReport apply(JsonArray rows, JsonObject angles);
    }

    /** Where the follower reports: Fiji's log and status bar, or a test. */
    interface Output {
        void line(String text);
        void status(String text);
    }

    static final Output FIJI = new Output() {
        public void line(String text) { FijiLog.line(text); }
        public void status(String text) { FijiLog.status(text); }
    };

    /** The applied-checkpoint log, inside the job folder. */
    static final String APPLIED_LOG = "logs/abba_applied.jsonl";
    private static final Map<MultiSlicePositioner, Follower> FOLLOWING = new WeakHashMap<>();

    final Path jobFolder;
    private final Target target;
    private final Output out;
    final WorkerClient client;
    private volatile boolean stopped;
    private volatile Thread thread;
    private int applied;
    /** The {@code state.json} modification time (ns) of the last {@code idle} event: ABBA shows the job as it was then. */
    private volatile long caughtUp = -1;
    /** Set for a follower of a live ABBA session (not in tests). */
    private MultiSlicePositioner mp;
    private AbbaHostSession host;
    private MultiSlicePositioner.MultiSlicePositionerListener closing;

    Follower(Path prefix, Path jobFolder, Target target, Output out) {
        this.jobFolder = jobFolder;
        this.target = target;
        this.out = out;
        this.client = new WorkerClient(prefix);
        this.client.stderrLog = out::line;
    }

    /**
     * Follows the job of *projectFolder* in *mp*: attaches (ABBA is brought back in line with the job by the first
     * checkpoint, one undo step) and starts the worker. A follower already on this job is kept unless some of its
     * rows were refused; then it is attached again, which is how a refused section gets retried.
     */
    static Follower ensure(MultiSlicePositioner mp, Path prefix, Path projectFolder) throws IOException {
        Path jobFolder = ProjectJob.jobFolder(projectFolder);
        Follower current;
        synchronized (FOLLOWING) { current = FOLLOWING.get(mp); }
        if (current != null && current.alive() && current.jobFolder.equals(jobFolder) && !current.host.hasPending()) return current;
        if (current != null) current.stop();
        AbbaHostSession host = new AbbaHostSession(mp, projectFolder, ProjectJob.folderKey(projectFolder));
        host.load();
        host.attach();
        Follower follower = new Follower(prefix, jobFolder, host::applyCheckpoint, FIJI);
        follower.mp = mp;
        follower.host = host;
        host.pendingReasons().forEach((id, reason) -> follower.out.line("LangSlice leaves " + id + " alone: " + reason));
        follower.closing = closed -> follower.stopInBackground();
        mp.addMultiSlicePositionerListener(follower.closing);
        synchronized (FOLLOWING) { FOLLOWING.put(mp, follower); }
        follower.start();
        follower.out.line("LangSlice follows " + jobFolder + ": every change the job saves appears in ABBA as one Undo step.");
        return follower;
    }

    /** Stops following in *mp*; false when nothing followed. ABBA keeps what it has. */
    static boolean stop(MultiSlicePositioner mp) {
        Follower current;
        synchronized (FOLLOWING) { current = FOLLOWING.remove(mp); }
        if (current == null) return false;
        current.stop();
        return true;
    }

    /** The follower of *mp*'s session, or null. */
    static Follower following(MultiSlicePositioner mp) {
        synchronized (FOLLOWING) { return FOLLOWING.get(mp); }
    }

    void start() {
        JsonObject params = new JsonObject();
        params.addProperty("job_folder", jobFolder.toString());
        Thread worker = new Thread(() -> {
            try {
                client.request("abba.follow", params, this::handleEvent, null);
            } catch (Exception failure) {
                if (!stopped) out.line("LangSlice stopped following " + jobFolder + ": " + WorkerClient.reason(failure)
                        + " Choose Register > LangSlice… in ABBA to follow it again.");
            } finally { ended(); }
        }, "LangSlice follower");
        worker.setDaemon(true);
        thread = worker;
        worker.start();
    }

    private void ended() {
        if (mp == null) return;
        synchronized (FOLLOWING) { if (FOLLOWING.get(mp) == this) FOLLOWING.remove(mp); }
        if (closing != null) {
            try { mp.removeMultiSlicePositionerListener(closing); } catch (RuntimeException closedAlready) { /* ABBA cleared it. */ }
        }
    }

    /** The {@code state.json} modification time (ns) ABBA has caught up to; -1 before the first {@code idle}. */
    long caughtUpTo() { return caughtUp; }

    boolean alive() { Thread running = thread; return running != null && running.isAlive() && !stopped; }

    /** Ends the worker (stdin closes; it is terminated if it does not finish) and waits for the follower to end. */
    void stop() {
        stopped = true;
        client.close();
        awaitEnd(WorkerClient.GRACE.plusSeconds(2));
    }

    /** ABBA's closing runs on its own thread: stopping waits for the worker, so it never holds ABBA up. */
    private void stopInBackground() {
        Thread stopper = new Thread(this::stop, "LangSlice follower stop");
        stopper.setDaemon(true);
        stopper.start();
    }

    boolean awaitEnd(Duration timeout) {
        Thread running = thread;
        if (running == null) return true;
        try { running.join(timeout.toMillis()); } catch (InterruptedException interrupted) { Thread.currentThread().interrupt(); }
        return !running.isAlive();
    }

    /** One streamed event: a checkpoint is applied, a log line goes to Fiji's log, "waiting" applies the attach. */
    void handleEvent(JsonObject event) {
        if (stopped) return;
        JsonObject payload = event.has("payload") && event.get("payload").isJsonObject() ? event.getAsJsonObject("payload") : event;
        String kind = payload.has("kind") && payload.get("kind").isJsonPrimitive() ? payload.get("kind").getAsString() : "";
        if (kind.equals("checkpoint")) {
            if (payload.has("initial") && payload.get("initial").getAsBoolean()) return;
            JsonArray rows = payload.has("host_updates") && payload.get("host_updates").isJsonArray() ? payload.getAsJsonArray("host_updates") : new JsonArray();
            JsonObject angles = payload.has("host_angles") && payload.get("host_angles").isJsonObject() ? payload.getAsJsonObject("host_angles") : null;
            applyLive(rows, angles);
        } else if (kind.equals("idle") && payload.has("state_mtime_ns")) {
            // Every checkpoint before this event has been applied (events are handled in order).
            caughtUp = payload.get("state_mtime_ns").getAsLong();
        } else if (kind.equals("log") && payload.has("message")) {
            out.line(payload.get("message").getAsString());
        } else if (kind.equals("waiting")) {
            out.line("LangSlice: the job in " + jobFolder + " has not started yet; ABBA follows it once it does.");
            // The job has no state: the attach's removals alone bring ABBA in line with it.
            applyLive(new JsonArray(), null);
        }
    }

    /** One checkpoint into ABBA; rows ABBA could not take are reported, logged and retried with the next checkpoint. */
    AbbaHostSession.ApplyReport applyLive(JsonArray rows, JsonObject angles) {
        AbbaHostSession.ApplyReport report;
        try { report = target.apply(rows, angles); }
        catch (RuntimeException failure) {
            // Never stop following over ABBA: the rows stay kept in the session and are retried next time.
            out.line("This step could not reach ABBA: " + failure.getMessage() + " It is retried with the next step.");
            return null;
        }
        if (!report.failed.isEmpty() || report.anglesFailed)
            out.line("Not in ABBA yet (retried with the next step; choose Register > LangSlice… to attach again): " + report.summary());
        if (report.changedAnything() || !report.failed.isEmpty() || report.anglesFailed) {
            applied++;
            try { AbbaHostSession.appendApplied(jobFolder.resolve(APPLIED_LOG), report); }
            catch (IOException unwritable) { out.line("Could not write " + APPLIED_LOG + ": " + unwritable.getMessage()); }
            out.status("LangSlice: " + applied + " step" + (applied == 1 ? "" : "s") + " applied to ABBA; each is one ABBA Undo.");
        }
        return report;
    }
}
