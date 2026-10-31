package org.langslice.fiji;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import java.util.prefs.Preferences;

/** Discover prefixes without requiring a terminal's activated conda environment. */
public final class EnvironmentDiscovery {
    private static final Preferences PREFS = Preferences.userNodeForPackage(EnvironmentDiscovery.class);
    /** Where conda and mamba installers put their base environment, relative to the home folder. */
    private static final List<String> CONDA_ROOTS = Arrays.asList("miniforge3", "miniconda3", "anaconda3", "mambaforge", "micromamba");
    private EnvironmentDiscovery() {}
    public static Path python(Path prefix) {
        Path windows = prefix.resolve("python.exe");
        return Files.isRegularFile(windows) ? windows : prefix.resolve("bin/python");
    }
    /** The environment the user saved in the environment dialog, or null. */
    public static Path saved() {
        String value = PREFS.get("environment", "");
        if (value.isEmpty()) return null;
        try { return Paths.get(value); } catch (InvalidPathException invalid) { return null; }
    }
    public static void save(Path prefix) { PREFS.put("environment", prefix.toAbsolutePath().normalize().toString()); }
    public static List<Path> discover() {
        Set<Path> candidates = new LinkedHashSet<>();
        Path remembered = saved();
        if (remembered != null) candidates.add(remembered);
        Path home = Paths.get(System.getProperty("user.home"));
        Path registry = home.resolve(".conda/environments.txt");
        try {
            for (String line : Files.readAllLines(registry, StandardCharsets.UTF_8)) {
                if (!line.trim().isEmpty() && !line.startsWith("#")) candidates.add(Paths.get(line.trim()));
            }
        } catch (IOException | InvalidPathException ignored) { /* Registry is optional. */ }
        for (Path root : condaRoots(home)) {
            Path envs = root.resolve("envs");
            candidates.add(envs.resolve("langslice"));
            try (DirectoryStream<Path> dirs = Files.newDirectoryStream(envs)) {
                for (Path prefix : dirs) candidates.add(prefix);
            } catch (IOException ignored) { }
        }
        candidates.add(home.resolve(".conda/envs/langslice"));
        List<Path> result = new ArrayList<>();
        for (Path path : candidates) if (Files.isRegularFile(python(path))) result.add(path.toAbsolutePath().normalize());
        final Path preferred = remembered == null ? null : remembered.toAbsolutePath().normalize();
        result.sort(Comparator.comparingInt((Path path) -> path.equals(preferred) ? 0 : "langslice".equalsIgnoreCase(path.getFileName().toString()) ? 1 : 2).thenComparing(Path::toString));
        return result;
    }
    /**
     * Conda base environments to search: the one named by CONDA_EXE or MAMBA_ROOT_PREFIX, then the installers'
     * default folders in the home folder.
     */
    static List<Path> condaRoots(Path home) {
        List<Path> roots = new ArrayList<>();
        String exe = System.getenv("CONDA_EXE");
        if (exe != null && !exe.isEmpty()) {
            try { Path base = Paths.get(exe).getParent(); if (base != null && base.getParent() != null) roots.add(base.getParent()); }
            catch (InvalidPathException ignored) { }
        }
        String mamba = System.getenv("MAMBA_ROOT_PREFIX");
        if (mamba != null && !mamba.isEmpty()) {
            try { roots.add(Paths.get(mamba)); } catch (InvalidPathException ignored) { }
        }
        for (String name : CONDA_ROOTS) roots.add(home.resolve(name));
        return roots;
    }
    /**
     * {@code python -u -m langslice ARGS} in the environment *prefix* (the worker: {@code serve --stdio}; the
     * window: {@code gui FOLDER}). conda run supplies activation variables, particularly DLL paths on Windows.
     */
    static List<String> command(Path prefix, String... args) {
        List<String> python = new ArrayList<>(Arrays.asList(python(prefix).toString(), "-u", "-m", "langslice"));
        python.addAll(Arrays.asList(args));
        if (!Files.isDirectory(prefix.resolve("conda-meta"))) return python;
        Path base = prefix.getParent() != null && "envs".equals(prefix.getParent().getFileName().toString())
                ? prefix.getParent().getParent() : prefix;
        Set<Path> executables = new LinkedHashSet<>();
        String configured = System.getenv("CONDA_EXE");
        if (configured != null && !configured.isEmpty()) executables.add(Paths.get(configured));
        List<Path> bases = new ArrayList<>();
        if (base != null) bases.add(base);
        Path home = Paths.get(System.getProperty("user.home"));
        bases.addAll(condaRoots(home));
        for (String variable : Arrays.asList("LOCALAPPDATA", "PROGRAMDATA")) {
            String root = System.getenv(variable);
            if (root != null) for (String install : CONDA_ROOTS) bases.add(Paths.get(root).resolve(install));
        }
        for (Path root : bases) for (String relative : Arrays.asList("bin/conda", "Scripts/conda.exe", "_conda.exe")) executables.add(root.resolve(relative));
        for (Path executable : executables) {
            if (Files.isRegularFile(executable)) {
                List<String> conda = new ArrayList<>(Arrays.asList(executable.toString(), "run", "--no-capture-output", "--prefix", prefix.toString()));
                conda.addAll(python);
                return conda;
            }
        }
        if (System.getProperty("os.name").toLowerCase(Locale.ROOT).contains("win"))
            throw new IllegalArgumentException("Conda could not be found to activate this environment. Install Miniforge in its standard location or set CONDA_EXE to the conda executable, then restart Fiji.");
        return python;
    }
}
