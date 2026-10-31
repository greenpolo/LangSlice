package org.langslice.fiji;

import ch.epfl.biop.atlas.aligner.MultiSlicePositioner;
import ch.epfl.biop.atlas.aligner.SliceSources;
import com.google.gson.JsonObject;
import org.scijava.command.Command;
import org.scijava.plugin.*;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.time.Duration;
import java.util.*;
import java.util.concurrent.TimeUnit;
import javax.swing.*;

/**
 * Register › LangSlice… (and Plugins › LangSlice › Open LangSlice…): the ABBA project's job is found or prepared, the
 * follower mirrors it into ABBA, and the LangSlice window opens on it: on the prompt generator for a job that has not
 * run, on the viewer otherwise. The registration itself runs in the user's agent app, from the generator's message.
 */
@Plugin(type = Command.class, name = LangSliceService.REGISTRATION, menuPath = "Plugins>LangSlice>Open LangSlice…")
public final class OpenLangSliceCommand implements Command {
    /** The worker protocol this connector speaks (setup.status, abba.prepare, abba.follow). */
    static final int PROTOCOL_VERSION = 3;
    /** A window process that ends sooner than this did not open. */
    static final Duration WINDOW_START = Duration.ofSeconds(3);

    @Parameter private MultiSlicePositioner mp;

    @Override public void run() { SwingUtilities.invokeLater(() -> EnvironmentDialog.ensureEnvironment(prefix -> open(mp, prefix))); }

    /** What the user chose when the project already has a job that differs from the selection. */
    enum Choice { CONTINUE, NEW, CANCEL }

    /** A message for the user rather than a failure report. */
    static final class Stop extends Exception {
        final boolean environment;
        Stop(String message, boolean environment) { super(message); this.environment = environment; }
    }

    static void open(MultiSlicePositioner mp, Path prefix) {
        new SwingWorker<String, Void>() {
            @Override protected String doInBackground() throws Exception { return openJob(mp, prefix); }
            @Override protected void done() {
                try { FijiLog.line(get()); }
                catch (Exception error) {
                    Throwable cause = error.getCause() == null ? error : error.getCause();
                    String message = cause instanceof Stop ? cause.getMessage() : "LangSlice could not open: " + WorkerClient.reason(cause);
                    FijiLog.line(message);
                    JOptionPane.showMessageDialog(null, message, "LangSlice", JOptionPane.WARNING_MESSAGE);
                    if (cause instanceof Stop && ((Stop) cause).environment) EnvironmentDialog.open();
                }
            }
        }.execute();
    }

    /** The whole flow, off the event thread; returns the line for Fiji's log. */
    static String openJob(MultiSlicePositioner mp, Path prefix) throws Exception {
        JsonObject status;
        try (WorkerClient worker = new WorkerClient(prefix)) {
            status = worker.request("setup.status", new JsonObject(), null, Duration.ofSeconds(120));
        } catch (Exception failed) {
            throw new Stop("LangSlice could not be started from " + prefix + ": " + WorkerClient.reason(failed)
                    + "\nChoose the environment that holds LangSlice.", true);
        }
        int protocol = status.has("protocol_version") ? status.get("protocol_version").getAsInt() : 0;
        if (protocol != PROTOCOL_VERSION)
            throw new Stop("LangSlice " + (status.has("version") ? status.get("version").getAsString() + " " : "")
                    + "in " + prefix + " speaks protocol " + protocol + "; this connector speaks " + PROTOCOL_VERSION
                    + ". Update LangSlice and the connector together.", false);
        AbbaHostSession.validateSession(mp);
        mp.waitForTasks();
        if (mp.getSlices().isEmpty()) throw new Stop("Import sections into ABBA before opening LangSlice.", false);
        List<SliceSources> selection = selection(mp);
        Set<String> chosen = names(selection);
        boolean selected = !mp.getSelectedSlices().isEmpty();
        ProjectJob.Found found = ProjectJob.find(mp);
        Path folder = found.folder;
        boolean prepare, hadJob = false;
        if (folder != null && ProjectJob.hasJob(folder)) {
            Set<String> job = ProjectJob.sectionNames(folder);
            Choice choice = job.isEmpty() || job.equals(chosen) ? Choice.CONTINUE : ask(job.size(), selection.size(), selected);
            if (choice == Choice.CANCEL) return "LangSlice: nothing changed.";
            prepare = choice == Choice.NEW;
            if (prepare) {
                Follower.stop(mp);
                Path aside = ProjectJob.startOver(folder);
                FijiLog.line("LangSlice: the earlier job of this project was moved to " + aside);
                hadJob = true;
            }
        } else if (folder != null) {
            prepare = true; // A folder whose preparation did not finish.
        } else if (!found.candidates.isEmpty()) {
            ProjectJob.Candidate best = found.candidates.get(0);
            Choice choice = ask(best.sections, selection.size(), selected);
            if (choice == Choice.CANCEL) return "LangSlice: nothing changed.";
            prepare = choice == Choice.NEW;
            if (prepare) folder = ProjectJob.root().resolve(found.key);
            else { folder = best.folder; ProjectJob.addKey(folder, found.key); }
        } else {
            prepare = true;
            folder = ProjectJob.root().resolve(found.key);
        }
        if (prepare) prepareJob(mp, prefix, folder, selection, hadJob);
        Follower.ensure(mp, prefix, folder);
        Path jobFolder = ProjectJob.jobFolder(folder);
        String tab = ProjectJob.hasCheckpoint(folder) ? "viewer" : "prompt";
        startWindow(prefix, folder, jobFolder, tab);
        return "LangSlice: " + (prepare ? "prepared the job " : "continued the job ") + jobFolder
                + "; the LangSlice window opens on the " + (tab.equals("prompt") ? "prompt generator." : "viewer.");
    }

    /** The selected slices, or every slice when none is selected, in atlas order. */
    static List<SliceSources> selection(MultiSlicePositioner mp) {
        List<SliceSources> selected = new ArrayList<>(mp.getSelectedSlices());
        if (selected.isEmpty()) selected.addAll(mp.getSlices());
        selected.sort(Comparator.comparingDouble(SliceSources::getSlicingAxisPosition));
        return selected;
    }

    static Set<String> names(List<SliceSources> slices) {
        Set<String> names = new HashSet<>();
        slices.forEach(slice -> names.add(slice.getName()));
        return names;
    }

    /** The one question of the flow, on the event thread. */
    static Choice ask(int jobSections, int sliceCount, boolean selected) throws Exception {
        String[] options = {
            "Continue the existing job (" + jobSections + " section" + (jobSections == 1 ? "" : "s") + ")",
            "Start a new job with " + (selected ? "the " + sliceCount + " selected" : "all " + sliceCount) + " slice" + (sliceCount == 1 ? "" : "s"),
        };
        int[] answer = {JOptionPane.CLOSED_OPTION};
        SwingUtilities.invokeAndWait(() -> answer[0] = JOptionPane.showOptionDialog(null,
                "This ABBA project already has a LangSlice job, on other sections than these.\n"
                        + "A new job sets the earlier one aside; its steps leave ABBA (one Undo brings them back).",
                "LangSlice", JOptionPane.DEFAULT_OPTION, JOptionPane.QUESTION_MESSAGE, null, options, options[0]));
        return answer[0] == 0 ? Choice.CONTINUE : answer[0] == 1 ? Choice.NEW : Choice.CANCEL;
    }

    /**
     * A new job in *folder*: this project's earlier LangSlice steps leave the sections (one undo step), the snapshots
     * are exported and the worker writes the prepared job.
     */
    static void prepareJob(MultiSlicePositioner mp, Path prefix, Path folder, List<SliceSources> selection, boolean hadJob) throws Exception {
        String key = ProjectJob.folderKey(folder);
        AbbaHostSession host = new AbbaHostSession(mp, folder, key);
        host.adopt(selection);
        host.attach();
        if (host.syncPending()) {
            AbbaHostSession.ApplyReport removed = host.applyCheckpoint(null, null);
            FijiLog.line("LangSlice removed this project's earlier steps from " + removed.synced.size() + " section(s) (one ABBA Undo step).");
        }
        int channels = Integer.MAX_VALUE;
        for (SliceSources slice : selection) channels = Math.min(channels, slice.getRegisteredSources().length);
        List<Integer> pages = new ArrayList<>();
        List<String> channelNames = new ArrayList<>();
        for (int c = 0; c < channels; c++) {
            pages.add(c);
            String name = selection.get(0).getRegisteredSources()[c].getSpimSource().getName();
            channelNames.add(name == null ? "" : name);
        }
        int[] protocol = ProjectJob.protocolDefaults(selection, hadJob);
        double pixelUm = host.snapshotPixelUm(selection, pages);
        FijiLog.line("LangSlice: exporting " + selection.size() + " calibrated snapshots at " + pixelUm + " µm per pixel to " + folder);
        JsonObject request = host.export(selection, pages, channelNames, pixelUm, protocol[1], protocol[0]);
        try (WorkerClient worker = new WorkerClient(prefix)) {
            worker.request("abba.prepare", request, null, Duration.ofMinutes(10));
        }
        List<String> all = new ArrayList<>();
        mp.getSlices().forEach(slice -> all.add(slice.getName()));
        ProjectJob.create(folder, mp.getAtlas().getName(), all, host.sectionNames());
    }

    /**
     * Starts {@code langslice gui JOB --tab TAB}; its output goes to {@code gui.log} in the project folder and its first
     * lines to Fiji's log (the window's address among them). A window that ends at once is reported with its output.
     */
    static Process startWindow(Path prefix, Path folder, Path jobFolder, String tab) throws Exception {
        Path log = folder.resolve("gui.log");
        Files.deleteIfExists(log);
        Files.createFile(log);
        Process window = WorkerClient.start(prefix, log, "gui", jobFolder.toString(), "--tab", tab);
        if (window.waitFor(WINDOW_START.toMillis(), TimeUnit.MILLISECONDS)) {
            String output = String.join("\n", firstLines(log, 12));
            throw new Stop("The LangSlice window did not open" + (output.isEmpty() ? "." : ":\n" + output)
                    + "\nCheck the environment in Plugins > LangSlice > Choose Python environment…", false);
        }
        Thread tail = new Thread(() -> forward(window, log), "LangSlice window log");
        tail.setDaemon(true);
        tail.start();
        return window;
    }

    static List<String> firstLines(Path log, int count) {
        try {
            List<String> lines = Files.readAllLines(log, StandardCharsets.UTF_8);
            return lines.subList(0, Math.min(count, lines.size()));
        } catch (IOException unreadable) { return Collections.emptyList(); }
    }

    /** Passes the window's first output lines (for a minute) to Fiji's log, so its address is there if no browser opened. */
    private static void forward(Process window, Path log) {
        int shown = 0;
        long deadline = System.nanoTime() + TimeUnit.MINUTES.toNanos(1);
        while (System.nanoTime() < deadline && shown < 20) {
            List<String> lines = firstLines(log, 20);
            for (; shown < lines.size(); shown++) if (!lines.get(shown).isBlank()) FijiLog.line("LangSlice window: " + lines.get(shown));
            if (!window.isAlive()) return;
            try { Thread.sleep(500); } catch (InterruptedException interrupted) { return; }
        }
    }
}
