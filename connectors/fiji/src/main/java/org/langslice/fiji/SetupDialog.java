package org.langslice.fiji;

import com.google.gson.*;
import java.awt.*;
import java.awt.event.*;
import java.net.URI;
import java.nio.file.*;
import java.time.Duration;
import java.util.Arrays;
import java.util.concurrent.CancellationException;
import javax.swing.*;

/** Modeless setup remains usable when LangSlice is not installed. */
public final class SetupDialog extends JDialog {
    private final JComboBox<String> environments = new JComboBox<>();
    private final JTextArea status = new JTextArea(8, 62);
    private final JButton browse = new JButton("Browse…");
    private Path verifiedPrefix;
    private final JButton check = new JButton("Check installation");
    private final JButton login = new JButton("Sign in with ChatGPT");
    private final JButton key = new JButton("Save API key…");
    private final JButton proceed = new JButton("Save setup");
    private final JButton cancel = new JButton("Cancel operation");
    private final JButton openLogin = new JButton("Open sign-in page");
    private final Runnable onReady;
    private WorkerClient worker;
    private boolean verified;
    private URI loginUri;
    /** The repository, whose README leads to the installation guide. */
    static final String INSTALLATION = "https://github.com/greenpolo/LangSlice";

    public static void open() { SwingUtilities.invokeLater(() -> new SetupDialog(null).setVisible(true)); }
    public static void ensureConfigured(Runnable ready) {
        // Recheck on each launch: an environment may have been moved or upgraded.
        SetupDialog dialog = new SetupDialog(ready);
        dialog.setVisible(true);
        dialog.checkInstallation(true);
    }
    private SetupDialog(Runnable onReady) {
        super((Frame) null, "LangSlice setup", false);
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
        environments.addActionListener(e -> invalidateVerification());
        ((JTextField) environments.getEditor().getEditorComponent()).getDocument().addDocumentListener(new javax.swing.event.DocumentListener() {
            public void insertUpdate(javax.swing.event.DocumentEvent e) { invalidateVerification(); }
            public void removeUpdate(javax.swing.event.DocumentEvent e) { invalidateVerification(); }
            public void changedUpdate(javax.swing.event.DocumentEvent e) { invalidateVerification(); }
        });
        status.setEditable(false); status.setLineWrap(true); status.setWrapStyleWord(true);
        status.setText("Choose the environment containing LangSlice and check the installation.\n\nIf LangSlice is not installed yet, follow the installation instructions, then reopen this dialog or browse to the new environment. You can install the Fiji plugin first.");
        JPanel actions = new JPanel(new FlowLayout(FlowLayout.LEFT));
        for (JButton button : Arrays.asList(check, login, key, openLogin, cancel, proceed)) actions.add(button);
        openLogin.setVisible(false); cancel.setEnabled(false); proceed.setEnabled(false);
        check.addActionListener(e -> checkInstallation(false));
        login.addActionListener(e -> { JsonObject params = new JsonObject(); params.addProperty("timeout_s", 300); execute("setup.login", params, false); });
        key.addActionListener(e -> saveKey());
        openLogin.addActionListener(e -> { try { Desktop.getDesktop().browse(loginUri); } catch (Exception failure) { JOptionPane.showMessageDialog(this, loginUri.toString(), "Open this address in your browser", JOptionPane.INFORMATION_MESSAGE); } });
        cancel.addActionListener(e -> { if (worker != null) worker.close(); });
        proceed.addActionListener(e -> { if (verified) { EnvironmentDiscovery.save(verifiedPrefix); dispose(); if (onReady != null) onReady.run(); } });
        JPanel root = new JPanel(new BorderLayout(10, 10)); root.setBorder(BorderFactory.createEmptyBorder(14, 14, 14, 14));
        root.add(location, BorderLayout.NORTH); root.add(new JScrollPane(status), BorderLayout.CENTER);
        JPanel bottom = new JPanel(new BorderLayout()); bottom.add(actions, BorderLayout.NORTH);
        JButton instructions = new JButton("Installation instructions");
        instructions.addActionListener(e -> {
            try { Desktop.getDesktop().browse(new URI(INSTALLATION)); }
            catch (Exception failure) { JOptionPane.showMessageDialog(this, "Open " + INSTALLATION + " in a browser."); }
        });
        bottom.add(instructions, BorderLayout.SOUTH); root.add(bottom, BorderLayout.SOUTH); setContentPane(root);
        addWindowListener(new WindowAdapter() { @Override public void windowClosed(WindowEvent e) { if (worker != null) worker.close(); } });
        pack(); setLocationByPlatform(true);
    }
    private void invalidateVerification() { verified = false; verifiedPrefix = null; proceed.setEnabled(false); }
    private Path prefix() {
        Object value = environments.getEditor().getItem();
        if (value == null || value.toString().trim().isEmpty()) throw new IllegalArgumentException("Choose the LangSlice environment folder first.");
        Path path = Paths.get(value.toString().trim()).toAbsolutePath().normalize();
        if (!Files.isRegularFile(EnvironmentDiscovery.python(path))) throw new IllegalArgumentException("Python was not found in this folder. Choose the conda environment folder, rather than a project folder or executable.");
        return path;
    }
    private void checkInstallation(boolean continueWhenReady) { execute("setup.status", new JsonObject(), continueWhenReady); }
    private void saveKey() {
        JComboBox<String> provider = new JComboBox<>(new String[]{"openai-api", "gemini-api"});
        JPasswordField password = new JPasswordField(32);
        JPanel form = new JPanel(new GridLayout(0, 1, 4, 4));
        form.add(new JLabel("Provider (API usage is billed by your provider):")); form.add(provider);
        form.add(new JLabel("API key:")); form.add(password);
        if (JOptionPane.showConfirmDialog(this, form, "Connect model provider", JOptionPane.OK_CANCEL_OPTION) != JOptionPane.OK_OPTION) return;
        char[] chars = password.getPassword();
        try {
            if (chars.length == 0) return;
            JsonObject params = new JsonObject(); params.addProperty("provider", provider.getSelectedItem().toString()); params.addProperty("api_key", new String(chars));
            execute("setup.api_key", params, false);
        } finally { Arrays.fill(chars, '\0'); password.setText(""); }
    }
    private void busy(boolean active) {
        check.setEnabled(!active); login.setEnabled(!active); key.setEnabled(!active); environments.setEnabled(!active); browse.setEnabled(!active); environments.getEditor().getEditorComponent().setEnabled(!active);
        cancel.setEnabled(active); proceed.setEnabled(!active && verified);
    }
    private void execute(String method, JsonObject params, boolean continueWhenReady) {
        final Path selected;
        try { selected = prefix(); } catch (RuntimeException failure) { status.setText(failure.getMessage()); return; }
        busy(true); status.setText("Working…"); openLogin.setVisible(false);
        worker = new WorkerClient(selected);
        final WorkerClient current = worker;
        new SwingWorker<JsonObject, Void>() {
            @Override protected JsonObject doInBackground() throws Exception {
                JsonObject result = current.request(method, params, event -> {
                    JsonObject payload = event.has("payload") && event.get("payload").isJsonObject() ? event.getAsJsonObject("payload") : event;
                    JsonElement url = payload.get("url");
                    if (url != null) SwingUtilities.invokeLater(() -> {
                        try { loginUri = URI.create(url.getAsString()); openLogin.setVisible(true); status.setText("Finish signing in in your browser. If no browser opened, click Open sign-in page."); pack(); }
                        catch (RuntimeException ignored) { }
                    });
                }, Duration.ofSeconds(method.equals("setup.login") ? 330 : 60));
                if (!method.equals("setup.status")) {
                    return current.request("setup.status", new JsonObject(), null, Duration.ofSeconds(60));
                }
                return result;
            }
            @Override protected void done() {
                try {
                    JsonObject result = get();
                    if (!isDisplayable()) return;
                    if (!result.has("protocol_version") || result.get("protocol_version").getAsInt() != 1) throw new IllegalStateException("This LangSlice version is incompatible with the plugin. Update LangSlice and the Fiji plugin together.");
                    verified = true; verifiedPrefix = selected;
                    StringBuilder text = new StringBuilder("Installation working. LangSlice ").append(result.get("version").getAsString()).append("\n\nAccounts:\n");
                    boolean configured = false;
                    if (result.has("providers")) for (java.util.Map.Entry<String, JsonElement> entry : result.getAsJsonObject("providers").entrySet()) {
                        JsonObject account = entry.getValue().getAsJsonObject();
                        if ("none".equals(entry.getKey())) continue;
                        boolean present = account.has("configured") && account.get("configured").getAsBoolean(); configured |= present;
                        text.append(entry.getKey()).append(present ? ": credentials saved (not validated online)\n" : ": not connected\n");
                    }
                    text.append("\nModel requests send selected section images to the chosen provider. API providers may charge for usage.");
                    status.setText(text.toString());
                    if (continueWhenReady && configured) {
                        // A check of the launcher's environment never replaces the user's own saved choice.
                        if (!selected.equals(EnvironmentDiscovery.launcher()) || EnvironmentDiscovery.saved() == null) EnvironmentDiscovery.save(selected);
                        dispose(); if (onReady != null) onReady.run();
                    }
                } catch (Exception failure) {
                    verified = false; Throwable cause = failure.getCause() != null ? failure.getCause() : failure;
                    status.setText(cause instanceof CancellationException ? "Operation cancelled." : cause.getMessage());
                } finally { params.remove("api_key"); busy(false); current.close(); worker = null; }
            }
        }.execute();
    }
}
