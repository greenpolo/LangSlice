package org.langslice.fiji;

import com.google.gson.*;
import java.awt.*;
import java.awt.event.*;
import java.awt.image.BufferedImage;
import java.util.*;
import java.util.List;
import javax.swing.*;
import javax.swing.table.DefaultTableModel;

/** The LangSlice Registration dialog. ABBA specifics sit behind Host, so the dialog renders without ABBA. */
final class RegistrationDialog extends JDialog {
    interface Host {
        /** Before (as exported) and after (what the agent sees) images of one listed slice. */
        BufferedImage[] preview(int slice, List<Integer> channels, JsonObject preprocessing, double pixelSize) throws Exception;
        JsonObject estimate(JsonObject params) throws Exception;
        JsonObject status() throws Exception;
        void setup();
        /**
         * Starts the run. nonlinearSkip: listed rows without an ABBA registration that Nonlinear leaves alone
         * (the user declined letting the agent align them first); empty otherwise.
         */
        void run(RegistrationSettings settings, Map<Integer, String> damaged, Set<Integer> nonlinearSkip);
        /** Whether LangSlice's agent viewer can open: ABBA was started from Python, which listens for run events. */
        default boolean viewerAvailable() { return LangSliceEvents.hasListeners(); }
        default void close() { }
    }

    static final class SliceRow {
        final String name; final int registrations;
        SliceRow(String name, int registrations) { this.name = name; this.registrations = registrations; }
    }

    static final String TIP = "Tip: try to maximize contrast between different regions.";
    static final String VIEWER_UNAVAILABLE = "Available when ABBA is started with `langslice abba`";
    private static final int PREVIEW_W = 340, PREVIEW_H = 230;
    private final Host host;
    private final List<SliceRow> rows;
    private final List<String> channelNames;
    private JsonObject status;
    private boolean statusStale, previewBusy, previewAgain, estimateBusy, estimateAgain, previewShown;

    final JComboBox<String> provider = new JComboBox<>(new String[]{"ChatGPT", "Claude"});
    final JLabel account = new JLabel();
    final JComboBox<String> model = new JComboBox<>();
    final JComboBox<RegistrationSettings.ImageChoice> imageModel = new JComboBox<>();
    final JComboBox<String> reasoning = new JComboBox<>(RegistrationSettings.REASONING);
    final JComboBox<String> resolution = new JComboBox<>(new String[]{"Low", "Medium", "High", "Auto"});
    final JCheckBox showLog = new JCheckBox("Show agent log"), viewer = new JCheckBox("Open agent viewer");
    final JCheckBox saveTraces = new JCheckBox("Save traces to");
    final JTextField traceDir = new JTextField(24);
    final JButton browseTraces = new JButton("Browse…");

    final JCheckBox positioning = header("Positioning"), flip = new JCheckBox("Enable hemisphere flipping");
    final JTextField cue = new JTextField(16);
    final JSpinner thickness = new JSpinner(new SpinnerNumberModel(0, 0, 100000, 10));
    final JSpinner interval = new JSpinner(new SpinnerNumberModel(0, 0, 100000, 10));
    final JTextArea positionNotes = notes();
    final JCheckBox linear = header("Linear"), affine = new JCheckBox("Enable affine tool");
    final JCheckBox angles = new JCheckBox("Enable slice angle estimation");
    final JSpinner parallel = new JSpinner(new SpinnerNumberModel(4, 1, 4, 1));
    final JTextArea linearNotes = notes();
    final JCheckBox nonlinear = header("Nonlinear");
    final JTextArea nonlinearNotes = notes();

    final DefaultTableModel table;
    final JCheckBox agentDamage = new JCheckBox("Let the agent mark damaged regions");
    final JCheckBox overwrite = new JCheckBox("Allow the agent to overwrite existing transforms");

    final JRadioButton auto = new JRadioButton("Auto"), custom = new JRadioButton("Custom");
    final JCheckBox agentPreprocessing = new JCheckBox("Let the agent drive preprocessing");
    final JSpinner[] weights;
    final JCheckBox clahe = new JCheckBox("CLAHE (local contrast)");
    final JComboBox<String> strength = new JComboBox<>(new String[]{"Low", "Medium", "High"});
    final JSpinner pixel = new JSpinner(new SpinnerNumberModel(25.0, 1.0, 1000.0, 5.0));
    final JComboBox<String> previewSlice = new JComboBox<>();
    final JLabel before = previewBox(), after = previewBox(), previewStatus = new JLabel(" ");
    final JButton refresh = new JButton("Update preview");

    final JTabbedPane tabs = new JTabbedPane();
    final JLabel cost = new JLabel("Estimated cost: calculating…");
    final JButton setup = new JButton("Setup…"), run = new JButton("Run"), cancel = new JButton("Cancel");
    private final javax.swing.Timer previewTimer = new javax.swing.Timer(700, e -> refreshPreview());
    private final javax.swing.Timer estimateTimer = new javax.swing.Timer(800, e -> refreshEstimate());

    RegistrationDialog(JsonObject status, List<SliceRow> rows, String caption, List<String> channelNames,
            RegistrationSettings settings, Host host) {
        super((Frame) null, "LangSlice Registration", false);
        this.status = status; this.rows = rows; this.channelNames = channelNames; this.host = host;
        setDefaultCloseOperation(DISPOSE_ON_CLOSE);
        previewTimer.setRepeats(false); estimateTimer.setRepeats(false);
        weights = new JSpinner[channelNames.size()];
        for (int c = 0; c < weights.length; c++) {
            weights[c] = new JSpinner(new SpinnerNumberModel(1.0, 0.0, 1.0, 0.1));
            weights[c].setEditor(new JSpinner.NumberEditor(weights[c], "0.0#"));
            ((JSpinner.DefaultEditor) weights[c].getEditor()).getTextField().setColumns(4);
        }
        table = new DefaultTableModel(new Object[]{"Slice", "Registrations", "Note on damaged tissue (optional)"}, 0) {
            @Override public Class<?> getColumnClass(int column) { return column == 1 ? Integer.class : String.class; }
            @Override public boolean isCellEditable(int row, int column) { return column == 2; }
        };
        for (SliceRow row : rows) { table.addRow(new Object[]{row.name, row.registrations, ""}); previewSlice.addItem(row.name); }
        fill(settings);

        tabs.addTab("Tasks", tasksTab());
        tabs.addTab("Slices", slicesTab(caption));
        tabs.addTab("Preprocessing", preprocessingTab());
        tabs.addChangeListener(e -> { if (tabs.getSelectedIndex() == 2 && !previewShown) { previewShown = true; refreshPreview(); } });
        JPanel root = new JPanel(new BorderLayout(0, 10));
        root.setBorder(BorderFactory.createEmptyBorder(12, 12, 12, 12));
        root.add(topPanel(), BorderLayout.NORTH);
        root.add(tabs, BorderLayout.CENTER);
        JPanel buttons = new JPanel(new FlowLayout(FlowLayout.RIGHT, 6, 0));
        buttons.add(setup); buttons.add(cancel); buttons.add(run);
        JPanel bottom = new JPanel(new BorderLayout(10, 0));
        bottom.add(cost, BorderLayout.CENTER); bottom.add(buttons, BorderLayout.EAST);
        root.add(bottom, BorderLayout.SOUTH);
        setContentPane(root);
        getRootPane().setDefaultButton(run);

        setup.addActionListener(e -> { statusStale = true; host.setup(); });
        cancel.addActionListener(e -> dispose());
        run.addActionListener(e -> run());
        refresh.addActionListener(e -> refreshPreview());
        addWindowListener(new WindowAdapter() {
            @Override public void windowActivated(WindowEvent e) { if (statusStale) { statusStale = false; refreshStatus(null); } }
            @Override public void windowClosed(WindowEvent e) { previewTimer.stop(); estimateTimer.stop(); host.close(); }
        });
        listen();
        sync();
        pack();
        setMinimumSize(getSize());
        setLocationByPlatform(true);
        estimateTimer.restart();
    }

    // ---- layout ---------------------------------------------------------------------------

    private JPanel topPanel() {
        JPanel top = new JPanel(new GridBagLayout());
        JPanel account = new JPanel(new FlowLayout(FlowLayout.LEFT, 6, 0));
        account.add(provider); account.add(this.account);
        provider.setToolTipText("Claude uses the LangSlice connector in Claude Desktop or Claude Code.");
        model.setEditable(true);
        model.setToolTipText("The model that runs the registration agent. You can type another model name.");
        imageModel.setToolTipText("The image model Nonlinear uses to trace region borders. None: deformations are fitted to the stain alone.");
        resolution.setToolTipText("How detailed the pictures the agent looks at are. Low is usually enough and costs least; Auto lets the agent choose each picture's size.");
        viewer.setToolTipText(host.viewerAvailable() ? "Show LangSlice's agent viewer: the agent's work in an ABBA-style display, as it happens." : VIEWER_UNAVAILABLE);
        cell(top, label("Provider"), 0, 0, 1, false); cell(top, account, 1, 0, 1, true);
        cell(top, label("Agent model"), 2, 0, 1, false); cell(top, model, 3, 0, 1, true);
        cell(top, label("Image resolution"), 0, 1, 1, false); cell(top, left(resolution), 1, 1, 1, true);
        cell(top, label("Reasoning"), 2, 1, 1, false); cell(top, reasoning, 3, 1, 1, true);
        cell(top, left(showLog, viewer), 0, 2, 2, true);
        cell(top, label("Image model"), 2, 2, 1, false); cell(top, imageModel, 3, 2, 1, true);
        saveTraces.setToolTipText("Keep a full record of the run: everything the agent was shown, said and did. Useful for reporting problems.");
        browseTraces.addActionListener(event -> {
            JFileChooser chooser = new JFileChooser(traceDir.getText().trim());
            chooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
            chooser.setDialogTitle("Folder for saved traces");
            if (chooser.showOpenDialog(this) == JFileChooser.APPROVE_OPTION) traceDir.setText(chooser.getSelectedFile().getPath());
        });
        JPanel traces = new JPanel(new BorderLayout(6, 0));
        traces.add(saveTraces, BorderLayout.WEST); traces.add(traceDir, BorderLayout.CENTER); traces.add(browseTraces, BorderLayout.EAST);
        cell(top, traces, 0, 3, 4, true);
        return top;
    }

    /** One grid for all three task sections, so their columns line up. */
    private JComponent tasksTab() {
        JPanel grid = new JPanel(new GridBagLayout());
        int y = 0;
        cell(grid, positioning, 0, y++, 4, true);
        cell(grid, indent(label("Section thickness (µm; 0 = infer)")), 0, y, 1, false); cell(grid, left(thickness), 1, y, 1, false);
        cell(grid, label("Section interval (µm; 0 = infer)"), 2, y, 1, false); cell(grid, left(interval), 3, y++, 1, false);
        cell(grid, indent(label("Notes for the agent")), 0, y, 1, false); cell(grid, scroll(positionNotes), 1, y++, 3, true);
        cell(grid, new JSeparator(), 0, y++, 4, true);

        cell(grid, linear, 0, y++, 4, true);
        cell(grid, indent(flip), 0, y, 2, false); cell(grid, label("Hemisphere cue"), 2, y, 1, false); cell(grid, cue, 3, y++, 1, true);
        cell(grid, indent(affine), 0, y, 2, false); cell(grid, label("Max parallel slice transforms"), 2, y, 1, false); cell(grid, left(parallel), 3, y++, 1, false);
        cell(grid, indent(angles), 0, y++, 4, true);
        cell(grid, indent(label("Notes for the agent")), 0, y, 1, false); cell(grid, scroll(linearNotes), 1, y++, 3, true);
        cell(grid, new JSeparator(), 0, y++, 4, true);
        flip.setToolTipText("The agent may mirror slices left-right; a mirror is part of the in-plane alignment.");
        cue.setToolTipText("Optional: how to tell left from right, for example 'ink mark on the right hemisphere'.");
        affine.setToolTipText("Off: the agent moves and scales each slice by hand only.");
        parallel.setToolTipText("How many slices the agent may transform in one step (1 = one slice at a time).");
        angles.setToolTipText("The agent may change the atlas cutting angles of the whole stack (ABBA's slicing angles).");

        cell(grid, nonlinear, 0, y++, 4, true);
        cell(grid, indent(label("Notes for the agent")), 0, y, 1, false); cell(grid, scroll(nonlinearNotes), 1, y++, 3, true);
        nonlinear.setToolTipText("A deformation per slice on top of its linear placement, added to ABBA as a BigWarp step.");
        return padded(grid);
    }

    private JComponent slicesTab(String caption) {
        JTable grid = new JTable(table);
        grid.setFillsViewportHeight(true);
        grid.setRowHeight(Math.max(grid.getRowHeight(), 22));
        grid.getColumnModel().getColumn(0).setPreferredWidth(260);
        grid.getColumnModel().getColumn(1).setPreferredWidth(90);
        grid.getColumnModel().getColumn(2).setPreferredWidth(310);
        JScrollPane pane = new JScrollPane(grid);
        pane.setPreferredSize(new Dimension(620, 230));
        long registered = rows.stream().filter(r -> r.registrations > 0).count();
        JPanel panel = new JPanel(new GridBagLayout());
        cell(panel, label(caption), 0, 0, 1, true);
        GridBagConstraints wide = constraints(0, 1, 1, true); wide.fill = GridBagConstraints.BOTH; wide.weighty = 1;
        panel.add(pane, wide);
        cell(panel, agentDamage, 0, 2, 1, true);
        agentDamage.setToolTipText("The agent may mark the atlas regions a slice has lost or displaced (mark_damage). Your notes are shown to it either way; a note alone does not mark a slice damaged.");
        cell(panel, overwrite, 0, 3, 1, true);
        cell(panel, indent(wrap(registered == 0 ? "No listed slice has an ABBA registration yet."
                : registered + " listed slice" + (registered == 1 ? " has" : "s have") + " ABBA registrations. With this option off, "
                + "their in-plane alignment is kept as it is; their positions can still move.")), 0, 4, 1, true);
        return padded(panel);
    }

    private JComponent preprocessingTab() {
        ButtonGroup group = new ButtonGroup(); group.add(auto); group.add(custom);
        JPanel controls = new JPanel(new GridBagLayout());
        cell(controls, label("Mode"), 0, 0, 1, false); cell(controls, left(auto, custom), 1, 0, 3, true);
        cell(controls, wrap("Auto weighs the channels and enhances contrast automatically. Custom uses the weights and contrast below."), 0, 1, 4, true);
        int y = 2;
        for (int c = 0; c < weights.length; c++, y++) {
            cell(controls, indent(label(channelLabel(c))), 0, y, 1, false); cell(controls, left(weights[c]), 1, y, 3, true);
            weights[c].setToolTipText("0 leaves this channel out.");
        }
        cell(controls, indent(clahe), 0, y, 1, false); cell(controls, left(label("Strength"), strength), 1, y++, 3, true);
        cell(controls, agentPreprocessing, 0, y++, 4, true);
        agentPreprocessing.setToolTipText("The agent may set channel weights, contrast and other steps for what it sees and what a fit reads. "
                + "Every channel is then sent, including those weighted 0.");
        cell(controls, label("Snapshot pixel size (µm)"), 0, y, 1, false); cell(controls, left(pixel), 1, y++, 3, true);
        pixel.setToolTipText("Size of one pixel in the images LangSlice exports from ABBA.");
        JLabel tip = label(TIP); tip.setFont(tip.getFont().deriveFont(Font.ITALIC));
        cell(controls, tip, 0, y, 4, true);

        JPanel preview = new JPanel(new GridBagLayout());
        cell(preview, label("Preview slice"), 0, 0, 1, false); cell(preview, previewSlice, 1, 0, 1, true); cell(preview, refresh, 2, 0, 1, false);
        JPanel images = new JPanel(new GridLayout(1, 2, 10, 0));
        images.add(captioned("Before: as exported from ABBA", before));
        images.add(captioned("After: what the agent sees", after));
        cell(preview, images, 0, 1, 3, true);
        cell(preview, previewStatus, 0, 2, 3, true);

        JPanel panel = new JPanel(new GridBagLayout());
        cell(panel, controls, 0, 0, 1, true);
        cell(panel, preview, 0, 1, 1, true);
        return padded(panel);
    }

    // ---- state ----------------------------------------------------------------------------

    private void fill(RegistrationSettings s) {
        provider.setSelectedItem(s.claude ? "Claude" : "ChatGPT");
        for (String id : RegistrationSettings.agentModels(status)) model.addItem(RegistrationSettings.modelLabel(id));
        fillImageModels(s.fresh ? RegistrationSettings.PROVIDER : s.imageProvider,
                s.fresh ? RegistrationSettings.accountDefault(status, "default_image_model") : s.imageModel);
        String agent = s.fresh ? RegistrationSettings.accountDefault(status, "default_agent_model") : s.model;
        if (agent != null && !agent.isEmpty()) select(model, RegistrationSettings.modelLabel(agent));
        reasoning.setSelectedItem(s.reasoning);
        resolution.setSelectedIndex(Math.max(0, Arrays.asList(RegistrationSettings.RESOLUTIONS).indexOf(s.resolution)));
        showLog.setSelected(s.showLog); viewer.setSelected(s.viewer && host.viewerAvailable());
        saveTraces.setSelected(s.saveTraces); traceDir.setText(s.traceDir);
        positioning.setSelected(s.positioning); flip.setSelected(s.flip); cue.setText(s.cue);
        thickness.setValue(s.thickness); interval.setValue(s.interval); positionNotes.setText(s.positionNotes);
        linear.setSelected(s.linear); affine.setSelected(s.affine); angles.setSelected(s.angles);
        parallel.setValue(s.maxParallel); linearNotes.setText(s.linearNotes);
        nonlinear.setSelected(s.nonlinear); nonlinearNotes.setText(s.nonlinearNotes);
        agentDamage.setSelected(s.agentDamage); overwrite.setSelected(s.overwrite);
        (s.custom ? custom : auto).setSelected(true);
        agentPreprocessing.setSelected(s.agentPreprocessing);
        double[] w = s.weightsFor(weights.length);
        for (int c = 0; c < weights.length; c++) weights[c].setValue(Math.max(0, Math.min(1, w[c])));
        clahe.setSelected(s.clahe);
        strength.setSelectedIndex(Math.max(0, Arrays.asList(RegistrationSettings.LEVELS).indexOf(s.strength)));
        pixel.setValue(s.pixelSize);
        updateAccount();
    }

    RegistrationSettings read() {
        RegistrationSettings s = new RegistrationSettings();
        s.claude = "Claude".equals(provider.getSelectedItem());
        Object typed = model.getEditor().getItem();
        s.model = RegistrationSettings.modelId(typed == null ? "" : typed.toString());
        RegistrationSettings.ImageChoice image = (RegistrationSettings.ImageChoice) imageModel.getSelectedItem();
        if (image != null) { s.imageProvider = image.provider; s.imageModel = image.model; }
        s.reasoning = String.valueOf(reasoning.getSelectedItem());
        s.resolution = RegistrationSettings.RESOLUTIONS[resolution.getSelectedIndex()];
        s.showLog = showLog.isSelected(); s.viewer = viewer.isSelected() && viewer.isEnabled();
        s.saveTraces = saveTraces.isSelected(); s.traceDir = traceDir.getText().trim();
        s.positioning = positioning.isSelected(); s.flip = flip.isSelected(); s.cue = cue.getText();
        s.thickness = (Integer) thickness.getValue(); s.interval = (Integer) interval.getValue(); s.positionNotes = positionNotes.getText();
        s.linear = linear.isSelected(); s.affine = affine.isSelected(); s.angles = angles.isSelected();
        s.maxParallel = (Integer) parallel.getValue(); s.linearNotes = linearNotes.getText();
        s.nonlinear = nonlinear.isSelected(); s.nonlinearNotes = nonlinearNotes.getText();
        s.agentDamage = agentDamage.isSelected(); s.overwrite = overwrite.isSelected();
        s.custom = custom.isSelected(); s.clahe = clahe.isSelected(); s.agentPreprocessing = agentPreprocessing.isSelected();
        s.strength = RegistrationSettings.LEVELS[strength.getSelectedIndex()];
        s.pixelSize = ((Number) pixel.getValue()).doubleValue();
        s.weights = new double[weights.length];
        for (int c = 0; c < weights.length; c++) s.weights[c] = ((Number) weights[c].getValue()).doubleValue();
        s.fresh = false;
        return s;
    }

    Map<Integer, String> damaged() {
        Map<Integer, String> marked = new LinkedHashMap<>();
        for (int r = 0; r < table.getRowCount(); r++)
            { String note = String.valueOf(table.getValueAt(r, 2)).trim(); if (!note.isEmpty()) marked.put(r, note); }
        return marked;
    }

    private void sync() {
        boolean p = positioning.isSelected(), l = linear.isSelected(), n = nonlinear.isSelected(), c = custom.isSelected();
        for (JComponent field : new JComponent[]{thickness, interval}) field.setEnabled(p);
        for (JComponent field : new JComponent[]{flip, affine, parallel, angles}) field.setEnabled(l);
        cue.setEnabled(l && flip.isSelected());
        enable(positionNotes, p); enable(linearNotes, l); enable(nonlinearNotes, n);
        for (JSpinner weight : weights) weight.setEnabled(c);
        clahe.setEnabled(c); strength.setEnabled(c && clahe.isSelected());
        viewer.setEnabled(host.viewerAvailable());
        traceDir.setEnabled(saveTraces.isSelected()); browseTraces.setEnabled(saveTraces.isSelected());
        boolean claude = "Claude".equals(provider.getSelectedItem());
        model.setEnabled(!claude); reasoning.setEnabled(!claude);
        // The image model belongs to Nonlinear, in both modes (Claude's job uses it through LangSlice).
        imageModel.setEnabled(n);
        run.setText(claude ? "Copy prompt" : "Run");
        updateAccount();
        run.setEnabled(p || l || n);
    }

    /** The image-model list from setup.status, keeping the given choice selected (added when the worker lists it no more). */
    private void fillImageModels(String provider, String model) {
        imageModel.removeAllItems();
        RegistrationSettings.ImageChoice chosen = null;
        for (RegistrationSettings.ImageChoice choice : RegistrationSettings.imageChoices(status)) {
            imageModel.addItem(choice);
            if (chosen == null && choice.matches(provider, model)) chosen = choice;
        }
        if (chosen == null && provider != null && !provider.isEmpty()) {
            chosen = new RegistrationSettings.ImageChoice(provider, model, provider + (model == null ? "" : ": " + model), true);
            imageModel.addItem(chosen);
        }
        if (chosen != null) imageModel.setSelectedItem(chosen);
    }

    private void listen() {
        ActionListener tasks = e -> { sync(); estimateTimer.restart(); };
        for (AbstractButton b : new AbstractButton[]{positioning, flip, linear, affine, angles, nonlinear, overwrite, agentDamage}) b.addActionListener(tasks);
        saveTraces.addActionListener(e -> sync());
        for (JComboBox<?> box : Arrays.asList(provider, model, reasoning, resolution, imageModel)) box.addActionListener(tasks);
        parallel.addChangeListener(e -> estimateTimer.restart());
        ActionListener prep = e -> { sync(); estimateTimer.restart(); schedulePreview(); };
        for (AbstractButton b : new AbstractButton[]{auto, custom, clahe, agentPreprocessing}) b.addActionListener(prep);
        strength.addActionListener(prep);
        previewSlice.addActionListener(e -> schedulePreview());
        for (JSpinner weight : weights) weight.addChangeListener(e -> schedulePreview());
        pixel.addChangeListener(e -> { estimateTimer.restart(); schedulePreview(); });
    }

    private void schedulePreview() { if (previewShown) previewTimer.restart(); }

    private void updateAccount() {
        if ("Claude".equals(provider.getSelectedItem())) {
            account.setText("Use Claude Desktop or Claude Code");
            account.setForeground(UIManager.getColor("Label.foreground")); return;
        }
        boolean in = RegistrationSettings.signedIn(status);
        account.setText(in ? "signed in" : "not signed in: use Setup…");
        account.setForeground(in ? UIManager.getColor("Label.foreground") : new Color(0xB00020));
    }

    // ---- actions --------------------------------------------------------------------------

    private void run() {
        RegistrationSettings s = read();
        String problem = s.problem(channelNames.size());
        if (problem != null) { JOptionPane.showMessageDialog(this, problem); return; }
        Set<Integer> skip = new LinkedHashSet<>();
        if (!askLinearFirst(s, skip)) return;
        if (s.claude || RegistrationSettings.signedIn(status)) { launch(s, skip); return; }
        refreshStatus(() -> {
            if (RegistrationSettings.signedIn(status)) launch(s, skip);
            else { JOptionPane.showMessageDialog(this, "Sign in with ChatGPT in LangSlice Setup first."); statusStale = true; host.setup(); }
        });
    }

    /** Listed rows without any ABBA registration: they have no linear placement for Nonlinear to build on. */
    List<Integer> unregistered() {
        List<Integer> missing = new ArrayList<>();
        for (int r = 0; r < rows.size(); r++) if (rows.get(r).registrations == 0) missing.add(r);
        return missing;
    }

    /**
     * Nonlinear needs a linear placement for every slice. With Linear off and some slices unregistered, ask:
     * yes switches Linear on; no leaves those slices out of Nonlinear (filled into skip). False: cancelled.
     */
    private boolean askLinearFirst(RegistrationSettings s, Set<Integer> skip) {
        List<Integer> missing = unregistered();
        if (!s.nonlinear || s.linear || missing.isEmpty()) return true;
        int answer = JOptionPane.showConfirmDialog(this, linearFirstQuestion(missing), "Nonlinear needs a linear step",
                JOptionPane.YES_NO_CANCEL_OPTION, JOptionPane.QUESTION_MESSAGE);
        if (answer == JOptionPane.YES_OPTION) { s.linear = true; linear.setSelected(true); sync(); return true; }
        if (answer == JOptionPane.NO_OPTION) {
            if (missing.size() == rows.size() && !s.positioning) {
                JOptionPane.showMessageDialog(this, "No listed slice has a linear registration, so Nonlinear has nothing to work on.");
                return false;
            }
            skip.addAll(missing); return true;
        }
        return false;
    }

    String linearFirstQuestion(List<Integer> missing) {
        StringBuilder names = new StringBuilder();
        for (int i = 0; i < missing.size() && i < 8; i++) names.append(i == 0 ? "" : ", ").append(rows.get(missing.get(i)).name);
        if (missing.size() > 8) names.append(" and ").append(missing.size() - 8).append(" more");
        return "<html><body width='420'>" + (missing.size() == 1 ? "Slice " : "Slices ") + escape(names.toString())
                + (missing.size() == 1 ? " has" : " have") + " no linear registration. Let the agent align "
                + (missing.size() == 1 ? "it" : "them") + " first?<br><br>Yes switches on Linear. No leaves "
                + (missing.size() == 1 ? "it" : "them") + " out of Nonlinear.</body></html>";
    }

    private static String escape(String text) { return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"); }

    private void launch(RegistrationSettings s, Set<Integer> skip) {
        s.save(RegistrationSettings.PREFS);
        Map<Integer, String> marked = damaged();
        dispose();
        host.run(s, marked, skip);
    }

    private void refreshStatus(Runnable then) {
        run.setEnabled(false);
        new SwingWorker<JsonObject, Void>() {
            @Override protected JsonObject doInBackground() throws Exception { return host.status(); }
            @Override protected void done() {
                try {
                    status = get();
                    RegistrationSettings.ImageChoice chosen = (RegistrationSettings.ImageChoice) imageModel.getSelectedItem();
                    if (chosen != null) fillImageModels(chosen.provider, chosen.model);
                } catch (Exception ignored) { }
                updateAccount(); sync();
                if (then != null && isDisplayable()) then.run();
            }
        }.execute();
    }

    void refreshPreview() {
        if (previewBusy) { previewAgain = true; return; }
        RegistrationSettings s = read();
        List<Integer> channels = s.exportChannels(channelNames.size());
        if (channels.isEmpty() || rows.isEmpty()) { previewStatus.setText("Give at least one channel a weight above 0."); return; }
        int index = Math.max(0, previewSlice.getSelectedIndex());
        JsonObject preprocessing = s.preprocessing(channelNames.size());
        previewBusy = true; refresh.setEnabled(false); previewStatus.setText("Preparing preview…");
        new SwingWorker<BufferedImage[], Void>() {
            @Override protected BufferedImage[] doInBackground() throws Exception { return host.preview(index, channels, preprocessing, s.pixelSize); }
            @Override protected void done() {
                previewBusy = false; refresh.setEnabled(true);
                try {
                    BufferedImage[] images = get();
                    show(before, images[0]); show(after, images[1]);
                    previewStatus.setText("Preview of " + rows.get(index).name + " at " + format(s.pixelSize) + " µm per pixel.");
                } catch (Exception failure) { previewStatus.setText("Preview unavailable: " + message(failure)); }
                if (previewAgain) { previewAgain = false; refreshPreview(); }
            }
        }.execute();
    }

    boolean previewBusy() { return previewBusy || previewTimer.isRunning(); }

    private void refreshEstimate() {
        if ("Claude".equals(provider.getSelectedItem())) { cost.setText("Usage is managed by Claude; no LangSlice estimate."); return; }
        RegistrationSettings s = read();
        String problem = s.problem(channelNames.size());
        if (problem != null) { cost.setText("Estimated cost: " + problem); return; }
        if (estimateBusy) { estimateAgain = true; return; }
        JsonObject params = new JsonObject();
        params.add("spec", s.spec());
        params.addProperty("n_slices", rows.size());
        params.addProperty("locked", s.overwrite ? 0 : rows.stream().filter(r -> r.registrations > 0).count());
        estimateBusy = true;
        new SwingWorker<JsonObject, Void>() {
            @Override protected JsonObject doInBackground() throws Exception { return host.estimate(params); }
            @Override protected void done() {
                estimateBusy = false;
                try {
                    JsonObject result = get();
                    if ("Claude".equals(provider.getSelectedItem())) { refreshEstimate(); return; }
                    cost.setText(costText(result));
                    cost.setToolTipText(result.has("basis") && result.get("basis").isJsonPrimitive() ? result.get("basis").getAsString() : null);
                } catch (Exception failure) {
                    Throwable cause = failure.getCause() != null ? failure.getCause() : failure;
                    cost.setText(refusalText(cause, s));
                    cost.setToolTipText(null);
                }
                if (estimateAgain) { estimateAgain = false; refreshEstimate(); }
            }
        }.execute();
    }

    /**
     * A plain reason when the worker gives no estimate for these settings: its own reason when it sends one,
     * otherwise what is known not to be measured (pictures above Low, Nonlinear).
     */
    static String refusalText(Throwable error, RegistrationSettings s) {
        String reason = workerReason(error);
        if (reason == null || reason.isEmpty()) {
            if (!"low".equals(s.resolution)) reason = "only runs at Low image resolution have been measured";
            else if (s.nonlinear) reason = "runs with Nonlinear have not been measured";
            else reason = "LangSlice could not estimate these settings";
        } else if (reason.startsWith("No runs have been measured at this image resolution")) {
            reason = "only runs at Low image resolution have been measured";
        }
        return "Estimated cost: no estimate (" + reason + ")";
    }

    /** The reason inside a worker error ({code, message, details: {error}}), or null. */
    static String workerReason(Throwable error) {
        String text = error == null ? null : error.getMessage();
        if (text == null) return null;
        try {
            JsonObject parsed = JsonParser.parseString(text).getAsJsonObject();
            if (parsed.has("details") && parsed.get("details").isJsonObject() && parsed.getAsJsonObject("details").has("error"))
                return parsed.getAsJsonObject("details").get("error").getAsString().trim();
            String message = parsed.has("message") ? parsed.get("message").getAsString() : null;
            return message == null || message.equals("Runtime request handling failed") ? null : message.trim();
        } catch (RuntimeException notJson) { return null; }
    }

    /** The estimate line; a worker that gives no number (available false, low/high null) gets its plain reason shown. */
    static String costText(JsonObject result) {
        boolean available = !result.has("available") || !result.get("available").isJsonPrimitive() || result.get("available").getAsBoolean();
        boolean numbers = result.has("low") && result.get("low").isJsonPrimitive() && result.has("high") && result.get("high").isJsonPrimitive();
        if (!available || !numbers) {
            String basis = result.has("basis") && result.get("basis").isJsonPrimitive() ? result.get("basis").getAsString().trim() : "";
            return "Estimated cost: no estimate" + (basis.isEmpty() ? "" : " (" + basis + ")");
        }
        double low = result.get("low").getAsDouble(), high = result.get("high").getAsDouble();
        String range = Math.abs(high - low) < 1e-9 ? format(low) : format(low) + "–" + format(high);
        String unit = result.has("unit") ? result.get("unit").getAsString() : "";
        return "Estimated cost: " + range + ("percent_of_usage_window".equals(unit) ? "% of your ChatGPT usage window" : " " + unit);
    }

    // ---- small helpers ------------------------------------------------------------------------

    private String channelLabel(int c) {
        String name = channelNames.get(c);
        if (name.length() > 28) name = name.substring(0, 27) + "…";
        return "Channel " + (c + 1) + (name.isEmpty() ? "" : ": " + name);
    }

    static String format(double value) {
        return Math.abs(value - Math.rint(value)) < 1e-9 ? String.valueOf((long) Math.rint(value)) : String.format(Locale.ROOT, "%.1f", value);
    }

    private static String message(Exception failure) {
        Throwable cause = failure.getCause() != null ? failure.getCause() : failure;
        return cause.getMessage() == null ? cause.getClass().getSimpleName() : cause.getMessage();
    }

    private static void select(JComboBox<String> box, String value) {
        for (int i = 0; i < box.getItemCount(); i++) if (box.getItemAt(i).equals(value)) { box.setSelectedIndex(i); return; }
        box.addItem(value); box.setSelectedItem(value);
    }

    private static void show(JLabel box, BufferedImage image) {
        if (image == null) { box.setIcon(null); box.setText("–"); return; }
        double scale = Math.min((double) (PREVIEW_W - 4) / image.getWidth(), (double) (PREVIEW_H - 4) / image.getHeight());
        int w = Math.max(1, (int) (image.getWidth() * scale)), h = Math.max(1, (int) (image.getHeight() * scale));
        box.setText(null);
        box.setIcon(new ImageIcon(image.getScaledInstance(w, h, Image.SCALE_SMOOTH)));
    }

    private static JLabel previewBox() {
        JLabel box = new JLabel("No preview yet", SwingConstants.CENTER);
        box.setPreferredSize(new Dimension(PREVIEW_W, PREVIEW_H));
        box.setOpaque(true); box.setBackground(Color.BLACK); box.setForeground(Color.LIGHT_GRAY);
        return box;
    }

    private static JPanel captioned(String caption, JComponent content) {
        JPanel panel = new JPanel(new BorderLayout(0, 4));
        panel.add(label(caption), BorderLayout.NORTH); panel.add(content, BorderLayout.CENTER);
        return panel;
    }

    private static JCheckBox header(String text) {
        JCheckBox box = new JCheckBox(text);
        box.setFont(box.getFont().deriveFont(Font.BOLD, box.getFont().getSize2D() + 1f));
        return box;
    }

    private static void enable(JTextArea area, boolean on) {
        area.setEnabled(on);
        Color off = UIManager.getColor("TextField.inactiveBackground");
        area.setBackground(on ? UIManager.getColor("TextArea.background") : off == null ? UIManager.getColor("Panel.background") : off);
    }

    private static JTextArea notes() {
        JTextArea area = new JTextArea(2, 30);
        area.setLineWrap(true); area.setWrapStyleWord(true);
        return area;
    }

    private static JScrollPane scroll(JTextArea area) {
        JScrollPane pane = new JScrollPane(area, ScrollPaneConstants.VERTICAL_SCROLLBAR_AS_NEEDED, ScrollPaneConstants.HORIZONTAL_SCROLLBAR_NEVER);
        pane.setPreferredSize(new Dimension(300, 44));
        pane.setMinimumSize(new Dimension(200, 44));
        return pane;
    }

    private static JLabel label(String text) { return new JLabel(text); }

    /** A wrapped explanatory line that follows the panel width. */
    private static JTextArea wrap(String text) {
        JTextArea area = new JTextArea(text);
        area.setEditable(false); area.setFocusable(false); area.setOpaque(false);
        area.setLineWrap(true); area.setWrapStyleWord(true);
        area.setFont(UIManager.getFont("Label.font")); area.setBorder(null);
        area.setColumns(52);
        return area;
    }

    private static JPanel left(JComponent... components) {
        JPanel row = new JPanel(new FlowLayout(FlowLayout.LEFT, 0, 0));
        for (int i = 0; i < components.length; i++) {
            if (i > 0) row.add(Box.createHorizontalStrut(14));
            row.add(components[i]);
        }
        return row;
    }

    private static JComponent indent(JComponent component) {
        JPanel row = new JPanel(new BorderLayout());
        row.setBorder(BorderFactory.createEmptyBorder(0, 22, 0, 0));
        row.add(component, BorderLayout.CENTER);
        return row;
    }

    private static JComponent padded(JPanel panel) {
        JPanel outer = new JPanel(new BorderLayout());
        outer.setBorder(BorderFactory.createEmptyBorder(8, 10, 8, 10));
        outer.add(panel, BorderLayout.NORTH);
        return outer;
    }

    private static GridBagConstraints constraints(int x, int y, int width, boolean stretch) {
        GridBagConstraints c = new GridBagConstraints();
        c.gridx = x; c.gridy = y; c.gridwidth = width;
        c.anchor = GridBagConstraints.WEST;
        c.fill = stretch ? GridBagConstraints.HORIZONTAL : GridBagConstraints.NONE;
        c.weightx = stretch ? 1 : 0;
        c.insets = new Insets(3, 4, 3, 8);
        return c;
    }

    private static void cell(JPanel panel, JComponent component, int x, int y, int width, boolean stretch) {
        panel.add(component, constraints(x, y, width, stretch));
    }
}
