package org.langslice.fiji;

import ch.epfl.biop.atlas.aligner.MultiSlicePositioner;
import ch.epfl.biop.atlas.aligner.SliceSources;
import com.google.gson.*;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.Instant;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;

/**
 * The stable LangSlice job of an ABBA project: {@code ~/.langslice/abba/<project key>/}, holding the snapshots
 * ({@code section_0001.tif ...}), the job folder {@code langslice/} beside them, {@link #PROJECT_FILE} and
 * {@link AbbaHostSession#SECTIONS_FILE}. ABBA keeps no accessor for its project file, so the key is computed from
 * the session: the atlas name and the names of ALL its slices.
 */
final class ProjectJob {
    /** What the folder is for: atlas, slice_names (all the session's), sections (snapshot to slice name), created_at, keys. */
    static final String PROJECT_FILE = "abba_project.json";
    /** The job folder inside the project folder (LangSlice's {@code JOB_DIRNAME}). */
    static final String JOB_DIRNAME = "langslice";
    private static final DateTimeFormatter STAMP = DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss", Locale.ROOT);
    /** A test's root instead of the home folder's. */
    private static volatile Path rootOverride;

    private ProjectJob() { }

    /** A project folder that may continue this session, and how many of the session's slice names it shares. */
    static final class Candidate {
        final Path folder;
        final int shared, sections;
        Candidate(Path folder, int shared, int sections) { this.folder = folder; this.shared = shared; this.sections = sections; }
        @Override public String toString() { return folder + " (" + shared + " shared slice names)"; }
    }

    /** What {@link #find} found: the key's own folder (null when there is none) and other folders sharing slice names. */
    static final class Found {
        final String key;
        final Path folder;
        final List<Candidate> candidates;
        Found(String key, Path folder, List<Candidate> candidates) { this.key = key; this.folder = folder; this.candidates = candidates; }
    }

    static Path root() {
        Path override = rootOverride;
        return override != null ? override : Paths.get(System.getProperty("user.home"), ".langslice", "abba");
    }

    static void useRoot(Path root) { rootOverride = root; }

    /** sha256(atlas name + "\n" + "\n".join(sorted slice names))[:12], in lower-case hex. */
    static String key(MultiSlicePositioner mp) {
        List<String> names = new ArrayList<>();
        for (SliceSources slice : mp.getSlices()) names.add(slice.getName());
        return key(mp.getAtlas().getName(), names);
    }

    static String key(String atlas, Collection<String> sliceNames) {
        List<String> sorted = new ArrayList<>(sliceNames);
        Collections.sort(sorted);
        try {
            byte[] digest = MessageDigest.getInstance("SHA-256").digest((atlas + "\n" + String.join("\n", sorted)).getBytes(StandardCharsets.UTF_8));
            StringBuilder hex = new StringBuilder();
            for (byte b : digest) hex.append(String.format(Locale.ROOT, "%02x", b));
            return hex.substring(0, 12);
        } catch (NoSuchAlgorithmException missing) { throw new IllegalStateException(missing); }
    }

    /** The session's project folder, else the folders whose slice names overlap the session's (most shared first). */
    static Found find(MultiSlicePositioner mp) {
        List<String> names = new ArrayList<>();
        for (SliceSources slice : mp.getSlices()) names.add(slice.getName());
        return find(mp.getAtlas().getName(), names);
    }

    static Found find(String atlas, Collection<String> sliceNames) {
        String key = key(atlas, sliceNames);
        Path own = root().resolve(key);
        if (Files.isDirectory(own)) return new Found(key, own, Collections.emptyList());
        Set<String> session = new HashSet<>(sliceNames);
        List<Candidate> candidates = new ArrayList<>();
        try (DirectoryStream<Path> folders = Files.newDirectoryStream(root())) {
            for (Path folder : folders) {
                // Only project folders: a started-over one (<key>.old-<time>) is kept aside, never continued.
                if (!folder.getFileName().toString().matches("[0-9a-f]{12}")) continue;
                JsonObject project = read(folder);
                if (project == null || !Files.isDirectory(folder) || !atlas.equals(string(project, "atlas"))) continue;
                // A folder this session already chose to continue (the session's slices changed since it was made).
                if (project.has("keys") && project.get("keys").isJsonArray())
                    for (JsonElement alias : project.getAsJsonArray("keys"))
                        if (key.equals(alias.getAsString())) return new Found(key, folder, Collections.emptyList());
                int shared = 0;
                if (project.has("slice_names") && project.get("slice_names").isJsonArray())
                    for (JsonElement name : project.getAsJsonArray("slice_names")) if (session.contains(name.getAsString())) shared++;
                int sections = project.has("sections") && project.get("sections").isJsonObject() ? project.getAsJsonObject("sections").size() : 0;
                if (shared > 0) candidates.add(new Candidate(folder, shared, sections));
            }
        } catch (IOException noRootYet) { /* Nothing made yet. */ }
        candidates.sort(Comparator.comparingInt((Candidate c) -> -c.shared).thenComparing(c -> c.folder.toString()));
        return new Found(key, null, candidates);
    }

    /** The project key a folder's steps carry: its own name (a continued folder keeps its key). */
    static String folderKey(Path folder) { return folder.getFileName().toString(); }

    static Path jobFolder(Path folder) { return folder.resolve(JOB_DIRNAME); }

    /** Whether the folder holds a prepared job (its {@code job.json}). */
    static boolean hasJob(Path folder) { return Files.isRegularFile(jobFolder(folder).resolve("job.json")); }

    /** Whether the job has run: its checkpoint {@code state.json} exists. */
    static boolean hasCheckpoint(Path folder) { return Files.isRegularFile(jobFolder(folder).resolve("state.json")); }

    /** The ABBA slice names of the folder's job sections ({@link AbbaHostSession#SECTIONS_FILE}), or an empty set. */
    static Set<String> sectionNames(Path folder) {
        Set<String> names = new HashSet<>();
        try {
            JsonObject mapping = JsonParser.parseString(Files.readString(folder.resolve(AbbaHostSession.SECTIONS_FILE), StandardCharsets.UTF_8)).getAsJsonObject();
            mapping.entrySet().forEach(entry -> names.add(entry.getValue().getAsString()));
        } catch (IOException | RuntimeException absent) { /* No job sections yet. */ }
        return names;
    }

    /** Writes {@link #PROJECT_FILE}: the atlas, all the session's slice names, the job's sections and when. */
    static void create(Path folder, String atlas, Collection<String> sliceNames, JsonObject sections) throws IOException {
        Files.createDirectories(folder);
        JsonObject project = new JsonObject();
        project.addProperty("atlas", atlas);
        JsonArray names = new JsonArray();
        List<String> sorted = new ArrayList<>(sliceNames);
        Collections.sort(sorted);
        sorted.forEach(names::add);
        project.add("slice_names", names);
        project.add("sections", sections);
        project.addProperty("created_at", Instant.now().toString());
        write(folder, project);
    }

    /** Records that the session with *key* continues *folder*, so the next {@link #find} goes straight to it. */
    static void addKey(Path folder, String key) throws IOException {
        JsonObject project = read(folder);
        if (project == null) project = new JsonObject();
        JsonArray keys = project.has("keys") && project.get("keys").isJsonArray() ? project.getAsJsonArray("keys") : new JsonArray();
        for (JsonElement known : keys) if (key.equals(known.getAsString())) return;
        keys.add(key);
        project.add("keys", keys);
        write(folder, project);
    }

    /** "Start a new job": the folder moves aside to {@code <folder>.old-<yyyyMMdd-HHmmss>}; returns where it went. */
    static Path startOver(Path folder) throws IOException {
        Path aside = folder.resolveSibling(folder.getFileName() + ".old-" + LocalDateTime.now().format(STAMP));
        for (int n = 2; Files.exists(aside); n++) aside = folder.resolveSibling(folder.getFileName() + ".old-" + LocalDateTime.now().format(STAMP) + "-" + n);
        return Files.move(folder, aside);
    }

    /**
     * Section interval and thickness in µm from ABBA's slices; 0 when ABBA gives no value. No interval when the
     * project folder already had a job: the positions may then be the agent's own placement.
     */
    static int[] protocolDefaults(List<SliceSources> slices, boolean hadJob) {
        List<Double> positions = new ArrayList<>(), thicknesses = new ArrayList<>();
        slices.forEach(slice -> {
            double p = slice.getSlicingAxisPosition(), t = slice.getThicknessInMm();
            if (Double.isFinite(p)) positions.add(p);
            if (Double.isFinite(t) && t > 0) thicknesses.add(t);
        });
        Collections.sort(positions);
        List<Double> gaps = new ArrayList<>();
        for (int i = 1; i < positions.size(); i++) if (positions.get(i) - positions.get(i - 1) > 1e-6) gaps.add(positions.get(i) - positions.get(i - 1));
        return new int[]{hadJob ? 0 : medianMicrons(gaps), medianMicrons(thicknesses)};
    }

    static int medianMicrons(List<Double> values) {
        if (values.isEmpty()) return 0;
        List<Double> sorted = new ArrayList<>(values);
        Collections.sort(sorted);
        int middle = sorted.size() / 2;
        double median = sorted.size() % 2 == 0 ? (sorted.get(middle - 1) + sorted.get(middle)) / 2 : sorted.get(middle);
        return Math.max(1, (int) Math.round(median * 1000));
    }

    private static JsonObject read(Path folder) {
        try { return JsonParser.parseString(Files.readString(folder.resolve(PROJECT_FILE), StandardCharsets.UTF_8)).getAsJsonObject(); }
        catch (IOException | RuntimeException absent) { return null; }
    }

    private static void write(Path folder, JsonObject project) throws IOException {
        Files.writeString(folder.resolve(PROJECT_FILE), new GsonBuilder().setPrettyPrinting().create().toJson(project), StandardCharsets.UTF_8);
    }

    private static String string(JsonObject object, String field) {
        return object.has(field) && object.get(field).isJsonPrimitive() ? object.get(field).getAsString() : null;
    }
}
