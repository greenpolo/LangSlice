package org.langslice.fiji;

import com.google.gson.JsonObject;
import java.awt.*;
import java.awt.event.*;
import java.net.URI;
import java.nio.file.*;
import java.time.Duration;
import java.util.Arrays;
import java.util.function.Consumer;
import javax.swing.*;

/** Chooses the Python environment that holds LangSlice: the connector's only dialog. Works without LangSlice installed. */
public final class EnvironmentDialog extends JDialog {
    /** The repository, whose README leads to the installation guide. */
    static final String INSTALLATION = "https://github.com/greenpolo/LangSlice";
    private final JComboBox<String> environments = new JComboBox<>();
    private final JTextArea status = new JTextArea(7, 60);
    private final JButton browse = new JButton("Browse…");
    private final JButton check = new JButton("Check installation");
    private final JButton save = new JButton("Save");
    private final Consumer<Path> onReady;
    private Path verified;
    private WorkerClient worker;

    /** Opens the dialog (Plugins › LangSlice › Choose Python environment…). */
    public static void open() { SwingUtilities.invokeLater(() -> new EnvironmentDialog(null).setVisible(true)); }

    /**
     * Calls *ready* with the saved environment, or opens the dialog when none is saved (or it has no Python) and
     * calls *ready* once the user saves one. Runs on the event thread.
     */
    static void ensureEnvironment(Consumer<Path> ready) {
        Path saved = EnvironmentDiscovery.saved();
        if (saved != null && Files.isRegularFile(EnvironmentDiscovery.python(saved))) { ready.accept(saved); return; }
        new EnvironmentDialog(ready).setVisible(true);
    }

    private EnvironmentDialog(Consumer<Path> onReady) {
        super((Frame) null, "LangSlice: Python environment", false);
        this.onReady = onReady;
        setDefaultCloseOperation(DISPOSE_ON_CLOSE);
        environments.setEditable(true);
        for (Path path : EnvironmentDiscovery.discover()) environments.addItem(path.toString());
        JPanel location = new JPanel(new BorderLayout(8, 8));
        location.add(new JLabel("LangSlice environment folder:"), BorderLayout.NORTH);
        location.add(environments, BorderLayout.CENTER);
        location.add(browse, BorderLayout.EAST);
        browse.addActionListener(e -> {
            JFileChooser chooser = new JFileChooser(); chooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
            if (chooser.showOpenDialog(this) == JFileChooser.APPROVE_OPTION) environments.setSelectedItem(chooser.getSelectedFile().getAbsolutePath());
        });
        environments.addActionListener(e -> verify(null));
        ((JTextField) environments.getEditor().getEditorComponent()).getDocument().addDocumentListener(new javax.swing.event.DocumentListener() {
            public void insertUpdate(javax.swing.event.DocumentEvent e) { verify(null); }
            public void removeUpdate(javax.swing.event.DocumentEvent e) { verify(null); }
            public void changedUpdate(javax.swing.event.DocumentEvent e) { verify(null); }
        });
        status.setEditable(false); status.setLineWrap(true); status.setWrapStyleWord(true);
        status.setText("Choose the environment that holds LangSlice, then check the installation.\n\nIf LangSlice is not installed yet, "
                + "follow the installation instructions, then come back here.");
        JPanel actions = new JPanel(new FlowLayout(FlowLayout.LEFT));
        JButton instructions = new JButton("Installation instructions");
        for (JButton button : Arrays.asList(check, save, instructions)) actions.add(button);
        save.setEnabled(false);
        check.addActionListener(e -> check());
        save.addActionListener(e -> {
            Path chosen = verified;
            if (chosen == null) return;
            EnvironmentDiscovery.save(chosen);
            dispose();
            if (onReady != null) onReady.accept(chosen);
        });
        instructions.addActionListener(e -> {
            try { Desktop.getDesktop().browse(new URI(INSTALLATION)); }
            catch (Exception failure) { JOptionPane.showMessageDialog(this, "Open " + INSTALLATION + " in a browser."); }
        });
        JPanel root = new JPanel(new BorderLayout(10, 10)); root.setBorder(BorderFactory.createEmptyBorder(14, 14, 14, 14));
        root.add(location, BorderLayout.NORTH); root.add(new JScrollPane(status), BorderLayout.CENTER); root.add(actions, BorderLayout.SOUTH);
        setContentPane(root);
        addWindowListener(new WindowAdapter() { @Override public void windowClosed(WindowEvent e) { WorkerClient running = worker; if (running != null) running.close(); } });
        pack(); setLocationByPlatform(true);
    }

    private void verify(Path chosen) { verified = chosen; save.setEnabled(chosen != null); }

    private Path prefix() {
        Object value = environments.getEditor().getItem();
        if (value == null || value.toString().trim().isEmpty()) throw new IllegalArgumentException("Choose the LangSlice environment folder first.");
        Path path = Paths.get(value.toString().trim()).toAbsolutePath().normalize();
        if (!Files.isRegularFile(EnvironmentDiscovery.python(path)))
            throw new IllegalArgumentException("Python was not found in this folder. Choose the conda environment folder, rather than a project folder or executable.");
        return path;
    }

    /** setup.status in the chosen environment: its LangSlice version and protocol. */
    private void check() {
        final Path chosen;
        try { chosen = prefix(); } catch (RuntimeException failure) { status.setText(failure.getMessage()); return; }
        check.setEnabled(false); verify(null); status.setText("Checking…");
        WorkerClient current = new WorkerClient(chosen);
        worker = current;
        new SwingWorker<JsonObject, Void>() {
            @Override protected JsonObject doInBackground() throws Exception {
                return current.request("setup.status", new JsonObject(), null, Duration.ofSeconds(120));
            }
            @Override protected void done() {
                try {
                    JsonObject result = get();
                    int protocol = result.has("protocol_version") ? result.get("protocol_version").getAsInt() : 0;
                    String version = result.has("version") ? result.get("version").getAsString() : "?";
                    if (protocol != OpenLangSliceCommand.PROTOCOL_VERSION) {
                        status.setText("LangSlice " + version + " is installed here, but it speaks protocol " + protocol + " and this connector speaks "
                                + OpenLangSliceCommand.PROTOCOL_VERSION + ". Update LangSlice and the connector together.");
                        return;
                    }
                    verify(chosen);
                    status.setText("LangSlice " + version + " works here.\nPython: "
                            + (result.has("python_executable") ? result.get("python_executable").getAsString() : chosen) + "\n\nSave to use this environment.");
                } catch (Exception failure) {
                    Throwable cause = failure.getCause() != null ? failure.getCause() : failure;
                    status.setText("LangSlice could not be started here: " + WorkerClient.reason(cause));
                } finally { check.setEnabled(true); current.close(); worker = null; }
            }
        }.execute();
    }
}
