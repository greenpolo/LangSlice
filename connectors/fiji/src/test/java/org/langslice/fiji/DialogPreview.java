package org.langslice.fiji;

import com.google.gson.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.*;
import java.util.List;
import javax.imageio.ImageIO;
import javax.swing.*;

/**
 * Optional visual QA, run with a display and no ABBA: DialogPreview OUTPUT_FOLDER.
 * Renders the Registration dialog's three tabs and both run windows from fake slices and a fake worker.
 */
public final class DialogPreview {
    public static void main(String[] args) throws Exception {
        File folder = new File(args.length > 0 ? args[0] : ".");
        JsonObject status = JsonParser.parseString("{\"protocol_version\":1,\"providers\":{\"openai-oauth\":{\"configured\":true,"
                + "\"agent_models\":[\"openai-oauth/gpt-6-astra\",\"openai-oauth/gpt-6-sol\",\"openai-oauth/gpt-6-luna\",\"openai-oauth/gpt-5.6-sol\",\"openai-oauth/gpt-5.6-terra\",\"openai-oauth/gpt-5.6-luna\"],"
                + "\"default_agent_model\":\"openai-oauth/gpt-5.6-sol\",\"image_models\":[\"gpt-image-2\"],\"default_image_model\":\"gpt-image-2\"},"
                + "\"gemini-api\":{\"configured\":true},\"openai-api\":{\"configured\":false},\"none\":{\"configured\":true}},"
                + "\"image_models\":[{\"provider\":\"openai-oauth\",\"label\":\"ChatGPT image lane\",\"connected\":true,\"models\":[\"gpt-image-2\"],\"default_model\":\"gpt-image-2\"},"
                + "{\"provider\":\"gemini-api\",\"label\":\"Gemini API\",\"connected\":true,\"models\":[\"gemini-3-pro-image\"],\"default_model\":\"gemini-3-pro-image\"},"
                + "{\"provider\":\"openai-api\",\"label\":\"OpenAI API\",\"connected\":false,\"models\":[\"gpt-image-2\"],\"default_model\":\"gpt-image-2\"},"
                + "{\"provider\":\"none\",\"label\":\"None\",\"connected\":true,\"models\":[],\"default_model\":null}]}").getAsJsonObject();
        List<RegistrationDialog.SliceRow> rows = new ArrayList<>();
        for (int i = 1; i <= 12; i++) rows.add(new RegistrationDialog.SliceRow("M03_B_" + String.format("%02d", i) + ".vsi - 10x_01", i % 5 == 0 ? 2 : 0));
        List<String> channels = Arrays.asList("DAPI", "NeuN-AF488", "Iba1-AF647 (a long channel name from the scanner)");
        RegistrationSettings settings = new RegistrationSettings();
        settings.interval = 120; settings.thickness = 40; settings.nonlinear = true; settings.angles = true;
        RegistrationDialog.Host host = new RegistrationDialog.Host() {
            public BufferedImage[] preview(int slice, List<Integer> pages, JsonObject preprocessing, double pixelSize) {
                return new BufferedImage[]{section(460, 320, true), section(460, 320, false)};
            }
            public JsonObject estimate(JsonObject params) {
                return JsonParser.parseString("{\"low\":3,\"high\":6,\"unit\":\"percent_of_usage_window\",\"basis\":\"12 slices, calibrated on recent runs\"}").getAsJsonObject();
            }
            public JsonObject status() { return status; }
            public void setup() { }
            public void run(RegistrationSettings s, Map<Integer, String> damaged, Set<Integer> skip) { }
            public boolean viewerAvailable() { return false; }
        };
        RegistrationDialog[] dialog = new RegistrationDialog[1];
        SwingUtilities.invokeAndWait(() -> {
            dialog[0] = new RegistrationDialog(status, rows, "12 slices selected in ABBA.", channels, settings, host);
            dialog[0].table.setValueAt("torn ventral cortex", 3, 2);
            dialog[0].custom.doClick(); dialog[0].weights[1].setValue(0.5); dialog[0].weights[2].setValue(0.0);
            dialog[0].setVisible(true);
        });
        settle(dialog[0]);
        String[] names = {"tasks", "slices", "preprocessing"};
        for (int tab = 0; tab < names.length; tab++) {
            final int index = tab;
            SwingUtilities.invokeAndWait(() -> dialog[0].tabs.setSelectedIndex(index));
            settle(dialog[0]);
            SwingUtilities.invokeAndWait(() -> capture(dialog[0], new File(folder, "dialog_" + names[index] + ".png")));
        }
        SwingUtilities.invokeAndWait(() -> dialog[0].dispose());
        for (boolean log : new boolean[]{true, false}) {
            RunWindow[] window = new RunWindow[1];
            SwingUtilities.invokeAndWait(() -> { window[0] = new RunWindow(log); window[0].open(); });
            window[0].line("Preparing calibrated snapshots in /tmp/langslice-abba-123");
            window[0].fragment("\nWorking: place slices\nThe first slices look like olfactory bulb; moving them anterior.\n");
            window[0].finish("Stopped. The changes the agent saved before that are in ABBA; each saved step is one ABBA Undo."
                    + " Some changes are not in ABBA: section_0004.tif (brain_04.vsi - 10x_01): its registrations were changed in ABBA during the run."
                    + " Use Retry failed updates to try again.");
            window[0].offerRetry(() -> { });
            Thread.sleep(400);
            SwingUtilities.invokeAndWait(() -> capture(window[0].frame, new File(folder, log ? "dialog_run_log.png" : "dialog_run_compact.png")));
            SwingUtilities.invokeAndWait(() -> window[0].frame.dispose());
        }
        System.exit(0);
    }

    private static void settle(RegistrationDialog dialog) throws Exception {
        for (int i = 0; i < 100; i++) {
            Thread.sleep(50);
            boolean[] busy = {false};
            SwingUtilities.invokeAndWait(() -> busy[0] = dialog.previewBusy() || dialog.cost.getText().contains("calculating"));
            if (!busy[0] && i > 20) break;
        }
    }

    private static void capture(Window window, File file) {
        try {
            BufferedImage image = new BufferedImage(window.getWidth(), window.getHeight(), BufferedImage.TYPE_INT_RGB);
            Graphics2D graphics = image.createGraphics();
            window.paint(graphics); graphics.dispose();
            ImageIO.write(image, "png", file);
            System.out.println(file + " " + window.getWidth() + "x" + window.getHeight());
        } catch (Exception failure) { throw new IllegalStateException(failure); }
    }

    /** A stand-in brain section: an ellipse with a few darker regions. */
    private static BufferedImage section(int w, int h, boolean colour) {
        BufferedImage image = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = image.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g.setColor(Color.BLACK); g.fillRect(0, 0, w, h);
        g.setColor(colour ? new Color(40, 150, 60) : new Color(150, 150, 150)); g.fillOval(40, 30, w - 80, h - 60);
        g.setColor(colour ? new Color(160, 40, 160) : new Color(90, 90, 90)); g.fillOval(w / 2 - 90, 80, 70, 110); g.fillOval(w / 2 + 20, 80, 70, 110);
        g.setColor(colour ? new Color(20, 80, 30) : new Color(210, 210, 210)); g.fillOval(w / 2 - 30, h / 2 + 30, 60, 40);
        g.dispose();
        return image;
    }
}
