package org.langslice.fiji;

import java.awt.*;
import javax.swing.*;

/** Run progress: the agent log, or only a status line. Stop, and an optional partial apply after a stop. */
final class RunWindow implements AgentRunner.Progress {
    final JFrame frame = new JFrame("LangSlice Registration");
    final JTextArea log;
    final JTextArea status = new JTextArea(3, 46);
    final JButton stop = new JButton("Stop run"), partial = new JButton("Apply partial result to ABBA"), close = new JButton("Close");

    RunWindow(boolean showLog) {
        status.setEditable(false); status.setFocusable(false); status.setOpaque(false);
        status.setLineWrap(true); status.setWrapStyleWord(true);
        status.setFont(UIManager.getFont("Label.font"));
        status.setText("Starting…");
        partial.setVisible(false); close.setEnabled(false);
        close.addActionListener(e -> frame.dispose());
        JPanel buttons = new JPanel(new FlowLayout(FlowLayout.RIGHT, 6, 0));
        buttons.add(partial); buttons.add(stop); buttons.add(close);
        JPanel root = new JPanel(new BorderLayout(0, 10));
        root.setBorder(BorderFactory.createEmptyBorder(12, 12, 12, 12));
        if (showLog) {
            log = new JTextArea(24, 80);
            log.setEditable(false); log.setLineWrap(true); log.setWrapStyleWord(true);
            root.add(status, BorderLayout.NORTH);
            root.add(new JScrollPane(log), BorderLayout.CENTER);
        } else {
            log = null;
            root.add(status, BorderLayout.CENTER);
        }
        root.add(buttons, BorderLayout.SOUTH);
        frame.setContentPane(root);
        frame.setDefaultCloseOperation(WindowConstants.HIDE_ON_CLOSE);
        frame.pack(); frame.setLocationByPlatform(true);
    }

    void open() { frame.setVisible(true); }

    @Override public void status(String text) { SwingUtilities.invokeLater(() -> status.setText(text)); }

    @Override public void line(String text) { fragment("\n" + text + "\n"); }

    @Override public void fragment(String text) {
        if (log == null) return;
        SwingUtilities.invokeLater(() -> {
            log.append(text);
            if (log.getDocument().getLength() > 200000) log.replaceRange("", 0, 50000);
            log.setCaretPosition(log.getDocument().getLength());
        });
    }

    /** The results are about to change ABBA: stopping is no longer offered. */
    void applying() { SwingUtilities.invokeLater(() -> stop.setEnabled(false)); }

    void finish(String message) {
        line(message);
        SwingUtilities.invokeLater(() -> {
            status.setText(message); stop.setEnabled(false); close.setEnabled(true);
            if (!frame.isVisible()) frame.setVisible(true);
        });
    }

    /** After a stop or failure: apply the last checkpoint once, on request. */
    void offerPartial(Runnable apply) {
        SwingUtilities.invokeLater(() -> {
            partial.setVisible(true);
            partial.addActionListener(e -> { partial.setEnabled(false); close.setEnabled(false); apply.run(); });
            frame.pack();
        });
    }

    void partialDone(String message) {
        finish(message);
        SwingUtilities.invokeLater(() -> partial.setVisible(false));
    }
}
