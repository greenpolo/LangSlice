package org.langslice.fiji;

import ch.epfl.biop.atlas.aligner.MultiSlicePositioner;
import com.google.gson.*;
import javax.swing.*;
import java.awt.*;
import java.nio.file.*;
import java.time.Duration;
import java.util.Collections;
import java.util.Set;
import java.util.WeakHashMap;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.atomic.AtomicBoolean;

/** Runs a separately installed Python engine while keeping the ABBA UI responsive. */
public final class AgentRunner {
    private static final Set<MultiSlicePositioner> RUNNING = Collections.newSetFromMap(new WeakHashMap<>());
    private AgentRunner() { }

    public static void show(MultiSlicePositioner mp, Path environment) {
        if (!SwingUtilities.isEventDispatchThread()) {
            SwingUtilities.invokeLater(() -> show(mp, environment)); return;
        }
        if (RUNNING.contains(mp)) {
            JOptionPane.showMessageDialog(null, "A LangSlice run is already active in this session."); return;
        }
        new SwingWorker<JsonObject,Void>() {
            protected JsonObject doInBackground() throws Exception {
                try (WorkerClient worker = new WorkerClient(environment)) {
                    return worker.request("setup.status",new JsonObject(),null,Duration.ofSeconds(60));
                }
            }
            protected void done() {
                try { showForm(mp,environment,get()); }
                catch (Exception error) { JOptionPane.showMessageDialog(null,"Could not check LangSlice. Open Setup to reconnect the environment."); SetupDialog.show(mp); }
            }
        }.execute();
    }

    private static void showForm(MultiSlicePositioner mp, Path environment, JsonObject status) {
        JComboBox<String> workflow = new JComboBox<>(new String[]{"Linear agent", "Nonlinear boundary refinement"});
        JCheckBox reorder = new JCheckBox("Order and orient sections", false);
        JCheckBox position = new JCheckBox("Estimate atlas positions", true);
        JCheckBox transform = new JCheckBox("Align sections in plane", true);
        JCheckBox flip = new JCheckBox("Allow left/right flips", true);
        JCheckBox strict = new JCheckBox("Use exact section spacing", false);
        JCheckBox interactive = new JCheckBox("Allow visual affine adjustment", true);
        JCheckBox automatic = new JCheckBox("Allow automatic affine fitting", true);
        JCheckBox elastix = new JCheckBox("Allow Elastix affine fitting", false);
        JComboBox<String> provider = new JComboBox<>(new String[]{"openai-oauth","gemini-api","openai-api"});
        JsonObject providers = status.getAsJsonObject("providers");
        for (String candidate : new String[]{"openai-oauth","gemini-api","openai-api"}) {
            if (providers.getAsJsonObject(candidate).get("configured").getAsBoolean()) { provider.setSelectedItem(candidate); break; }
        }
        JTextField model = new JTextField(defaultModel((String)provider.getSelectedItem()));
        provider.addActionListener(event -> model.setText(workflow.getSelectedIndex()==0 ? defaultModel((String)provider.getSelectedItem()) : "provider default"));
        JComboBox<String> reasoning = new JComboBox<>(new String[]{"default","low","medium","high","xhigh","max"});
        JTextField cue = new JTextField();
        JTextArea facts = new JTextArea(3, 30);
        JSpinner interval = new JSpinner(new SpinnerNumberModel(protocolDefaults(mp)[0], 1, 100000, 10));
        JSpinner thickness = new JSpinner(new SpinnerNumberModel(protocolDefaults(mp)[1], 1, 100000, 10));
        JSpinner pixel = new JSpinner(new SpinnerNumberModel(25.0, 1.0, 1000.0, 5.0));
        JSpinner channel = new JSpinner(new SpinnerNumberModel(1, 1, 100, 1));
        JPanel form = new JPanel(new GridLayout(0,2,8,5));
        form.add(new JLabel("Applies to selected sections, or all if none selected.")); form.add(new JLabel());
        row(form,"Method",workflow);
        form.add(reorder); form.add(position); form.add(transform); form.add(flip);
        row(form,"Model account",provider); row(form,"Model",model); row(form,"Reasoning",reasoning);
        row(form,"Section interval (µm)",interval); row(form,"Section thickness (µm)",thickness);
        form.add(strict); form.add(interactive); form.add(automatic); form.add(elastix);
        row(form,"Hemisphere cue",cue); row(form,"Additional facts (one per line)",new JScrollPane(facts));
        row(form,"Image channel (1 = first)",channel); row(form,"Snapshot pixel size (µm)",pixel);
        form.add(new JLabel("Images are sent to the selected model provider.")); form.add(new JLabel());
        workflow.addActionListener(event -> {
            boolean linear = workflow.getSelectedIndex()==0;
            for (JComponent field : new JComponent[]{reorder,position,transform,flip,strict,interactive,automatic,elastix,cue,facts,interval,thickness,reasoning}) field.setEnabled(linear);
            model.setText(linear ? defaultModel((String)provider.getSelectedItem()) : "provider default");
        });
        if (JOptionPane.showConfirmDialog(null,form,"LangSlice agent",JOptionPane.OK_CANCEL_OPTION,JOptionPane.PLAIN_MESSAGE)!=JOptionPane.OK_OPTION) return;
        JsonArray tasks = new JsonArray();
        if (reorder.isSelected()) tasks.add("reorder"); if (position.isSelected()) tasks.add("position"); if (transform.isSelected()) tasks.add("transform");
        if (workflow.getSelectedIndex()==0 && tasks.size()==0) { JOptionPane.showMessageDialog(null,"Select at least one task."); return; }
        if (model.getText().trim().isEmpty()) { JOptionPane.showMessageDialog(null,"Choose a model before starting."); return; }
        if (workflow.getSelectedIndex()==0 && transform.isSelected() && !interactive.isSelected() && !automatic.isSelected()) {
            JOptionPane.showMessageDialog(null,"Enable visual adjustment or automatic fitting."); return;
        }
        if (!providers.getAsJsonObject((String)provider.getSelectedItem()).get("configured").getAsBoolean()) {
            JOptionPane.showMessageDialog(null,"Connect the selected model account in LangSlice Setup first.");
            SetupDialog.show(mp); return;
        }
        JsonObject spec = new JsonObject(); spec.add("tasks",tasks); spec.addProperty("model",model.getText().trim());
        if (reasoning.getSelectedIndex()>0) spec.addProperty("reasoning",(String)reasoning.getSelectedItem());
        JsonObject ordering = new JsonObject(); ordering.addProperty("flip",flip.isSelected()); ordering.addProperty("hemisphere_cue",cue.getText()); spec.add("reorder",ordering);
        JsonObject positioning = new JsonObject(); positioning.addProperty("interval_um",(Integer)interval.getValue()); positioning.addProperty("thickness_um",(Integer)thickness.getValue()); positioning.addProperty("strict_interval",strict.isSelected()); spec.add("position",positioning);
        JsonObject alignment = new JsonObject(); alignment.addProperty("interactive",interactive.isSelected()); alignment.addProperty("automatic",automatic.isSelected()); alignment.addProperty("elastix",elastix.isSelected()); alignment.addProperty("angles",false); spec.add("transform",alignment);
        JsonArray knowledge = new JsonArray(); for (String line : facts.getText().split("\\R")) if (!line.trim().isEmpty()) knowledge.add(line.trim()); spec.add("facts",knowledge);
        if (workflow.getSelectedIndex()==1) {
            spec = new JsonObject(); spec.addProperty("provider",(String)provider.getSelectedItem());
            spec.addProperty("review_model",defaultModel((String)provider.getSelectedItem()));
            if (!model.getText().trim().equals("provider default")) spec.addProperty("model",model.getText().trim());
        }
        start(mp,environment,spec,workflow.getSelectedIndex()==1,(Integer)channel.getValue()-1,((Number)pixel.getValue()).doubleValue());
    }

    private static String defaultModel(String provider) {
        if ("gemini-api".equals(provider)) return "gemini-3-flash-preview";
        if ("openai-api".equals(provider)) return "gpt-5.4";
        return "openai-oauth/gpt-5.6-sol";
    }

    static int[] protocolDefaults(MultiSlicePositioner mp) {
        java.util.List<Double> positions = new java.util.ArrayList<>();
        java.util.List<Double> thicknesses = new java.util.ArrayList<>();
        mp.getSlices().forEach(slice -> {
            double p=slice.getSlicingAxisPosition(), t=slice.getThicknessInMm();
            if (Double.isFinite(p)) positions.add(p);
            if (Double.isFinite(t) && t>0) thicknesses.add(t);
        });
        Collections.sort(positions); java.util.List<Double> gaps = new java.util.ArrayList<>();
        for (int i=1;i<positions.size();i++) if (positions.get(i)-positions.get(i-1)>1e-6) gaps.add(positions.get(i)-positions.get(i-1));
        return new int[]{medianMicrons(gaps,200),medianMicrons(thicknesses,50)};
    }

    static int medianMicrons(java.util.List<Double> values,int fallback) {
        if (values.isEmpty()) return fallback;
        Collections.sort(values); int middle=values.size()/2;
        double median=values.size()%2==0 ? (values.get(middle-1)+values.get(middle))/2 : values.get(middle);
        return Math.max(1,(int)Math.round(median*1000));
    }

    private static void row(JPanel form,String title,JComponent component) { form.add(new JLabel(title)); form.add(component); }

    private static void start(MultiSlicePositioner mp,Path environment,JsonObject spec,boolean nonlinear,int channel,double pixelSize) {
        if (RUNNING.contains(mp)) { JOptionPane.showMessageDialog(null,"A LangSlice run is already active in this session."); return; }
        RUNNING.add(mp);
        JFrame window = new JFrame("LangSlice agent activity");
        JTextArea log = new JTextArea(24,80); log.setEditable(false); log.setLineWrap(true); log.setWrapStyleWord(true);
        JButton cancel = new JButton("Stop run");
        window.add(new JScrollPane(log),BorderLayout.CENTER); window.add(cancel,BorderLayout.SOUTH);
        window.setDefaultCloseOperation(WindowConstants.HIDE_ON_CLOSE); window.pack(); window.setLocationByPlatform(true); window.setVisible(true);
        AtomicReference<WorkerClient> client = new AtomicReference<>();
        AtomicReference<Thread> runningThread = new AtomicReference<>();
        AtomicBoolean stopping = new AtomicBoolean();
        SwingWorker<Void,Void> worker = new SwingWorker<Void,Void>() {
            protected Void doInBackground() throws Exception {
                runningThread.set(Thread.currentThread());
                if (stopping.get()) throw new java.util.concurrent.CancellationException();
                Path folder = Files.createTempDirectory("langslice-abba-");
                append(log,"Preparing calibrated snapshots in " + folder);
                append(log,"Keep ABBA registrations unchanged while this run is active.");
                AbbaHostSession host = new AbbaHostSession(mp,folder);
                if (nonlinear) {
                    host.nonlinear(environment,spec,channel,pixelSize,
                        event -> displayEvent(log,event),
                        client::set,stopping);
                    append(log,"Nonlinear registration completed. Save the project in ABBA. Run files: " + folder);
                    return null;
                }
                JsonObject request = host.prepare(spec,channel,pixelSize);
                AbbaHostSession.checkInterrupted();
                try (WorkerClient connection = new WorkerClient(environment)) {
                    client.set(connection);
                    JsonObject result = connection.request("linear.run",request,event -> {
                        if (stopping.get()) throw new java.util.concurrent.CancellationException();
                        if (event.has("payload")) {
                            JsonObject payload = event.getAsJsonObject("payload");
                            String kind = payload.has("kind") ? payload.get("kind").getAsString() : "";
                            if (kind.equals("checkpoint")) {
                                host.apply(payload.getAsJsonArray("host_updates"));
                                append(log,"ABBA updated from agent checkpoint.");
                            } else if (kind.equals("log") && payload.has("message")) append(log,payload.get("message").getAsString());
                            else if (kind.equals("agent_event")) displayAgentEvent(log,payload.getAsJsonObject("event"));
                        } else if (event.has("message")) append(log,event.get("message").getAsString());
                    },Duration.ofHours(12));
                    append(log,(result.getAsJsonObject("state").get("submitted").getAsBoolean() ? "Completed and submitted. " : "Stopped before submission; review the partial results. ") + "Save the project using ABBA's normal Save command. Run files: " + result.get("output_dir").getAsString());
                } finally { client.set(null); }
                return null;
            }
            protected void done() {
                RUNNING.remove(mp); cancel.setEnabled(false);
                try { get(); }
                catch (java.util.concurrent.CancellationException e) { append(log,"Stopped. Completed changes remain in ABBA and can be undone."); }
                catch (Exception e) { Throwable cause=e.getCause()==null?e:e.getCause(); append(log,stopping.get() ? "Stopped. Completed changes remain in ABBA and can be undone." : "Run stopped: " + cause.getMessage()); }
            }
        };
        cancel.addActionListener(event -> {
            stopping.set(true); cancel.setEnabled(false);
            WorkerClient connection=client.get(); if (connection!=null) connection.close();
            Thread thread=runningThread.get(); if (connection==null && thread!=null) thread.interrupt();
        });
        worker.execute();
    }

    private static void displayEvent(JTextArea log, JsonObject event) {
        JsonObject payload = event.has("payload") ? event.getAsJsonObject("payload") : event;
        if (payload.has("message")) append(log,payload.get("message").getAsString());
        if (payload.has("event")) displayAgentEvent(log,payload.getAsJsonObject("event"));
    }

    static String agentText(JsonObject event) {
        String kind = event.has("kind") ? event.get("kind").getAsString() : "";
        if ((kind.equals("text") || kind.equals("reasoning")) && event.has("text")) return event.get("text").getAsString();
        if (kind.equals("tool_start")) return "\nWorking: " + (event.has("name") ? event.get("name").getAsString().replace('_',' ') : "registration") + "\n";
        if (kind.equals("tool_result") && event.has("response") && event.get("response").isJsonObject()) {
            JsonObject response=event.getAsJsonObject("response");
            String status=response.has("status") ? response.get("status").getAsString() : "";
            if (!status.isEmpty() && !status.equals("ok") && !status.equals("success")) {
                String reason=response.has("message") ? response.get("message").getAsString() : response.has("reason") ? response.get("reason").getAsString() : status;
                return "\n" + reason.substring(0,Math.min(500,reason.length())) + "\n";
            }
        }
        if (kind.equals("error") && event.has("text")) return "\nRun failed: " + event.get("text").getAsString() + "\n";
        return "";
    }

    private static void displayAgentEvent(JTextArea log, JsonObject event) {
        String text=agentText(event);
        if (!text.isEmpty()) appendFragment(log,text);
    }

    private static void appendFragment(JTextArea area,String message) {
        SwingUtilities.invokeLater(() -> {
            area.append(message);
            if (area.getDocument().getLength()>200000) area.replaceRange("",0,50000);
            area.setCaretPosition(area.getDocument().getLength());
        });
    }

    private static void append(JTextArea area,String message) {
        appendFragment(area,"\n"+message+"\n");
    }
}
