package org.langslice.fiji;

import ch.epfl.biop.atlas.aligner.MultiSlicePositioner;
import ch.epfl.biop.atlas.aligner.SliceSources;
import com.google.gson.*;
import net.imglib2.realtransform.AffineTransform3D;
import java.nio.file.*;

/**
 * Opt-in cached-atlas integration smoke; never contacts a model provider.
 * After build.py --test, run with the same resolved test classpath:
 * java -Djava.awt.headless=true -cp TEST_CLASSES:CLASSES:MAVEN_CLASSPATH
 *     org.langslice.fiji.HostSessionSmokeTest /path/to/existing/cached_atlas
 * Requires the Allen V3p1 XML, HDF5 and ontology already downloaded by ABBA.
 * The test creates its own calibrated synthetic TIFF and fresh temporary project.
 */
public final class HostSessionSmokeTest {
    public static void main(String[] args) throws Exception {
        if (args.length != 1) throw new IllegalArgumentException("Supply an existing ABBA Allen V3p1 atlas cache folder");
        Path cache=Paths.get(args[0]);
        for (String name : new String[]{"mouse_brain_ccfv3p1.xml","ccf2017-mod65000-border-centered-mm-bc.h5","1.json"})
            require(Files.isRegularFile(cache.resolve(name)),"Missing cached atlas file: "+name);
        ch.epfl.biop.atlas.AtlasLocationHelper.defaultCacheDir=cache.toFile();
        Path output=Files.createTempDirectory("langslice-host-smoke-");
        net.imagej.ImageJ imagej=new net.imagej.ImageJ();
        Object atlas=imagej.command().run(ch.epfl.biop.atlas.mouse.allen.ccfv3p1.command.AllenBrainAdultMouseAtlasCCF2017v3p1Command.class,true).get().getOutput("ba");
        MultiSlicePositioner mp=(MultiSlicePositioner)imagej.command().run(ch.epfl.biop.atlas.aligner.command.ABBAStartCommand.class,true,
                "x_axis","RL","y_axis","SI","z_axis","AP","ba",atlas).get().getOutput("mp");
        ij.ImagePlus image=ij.IJ.createImage("Synthetic section","8-bit ramp",128,96,1);
        image.getCalibration().pixelWidth=25;image.getCalibration().pixelHeight=25;image.getCalibration().setUnit("um");
        new ij.io.FileSaver(image).saveAsTiff(output.resolve("synthetic.tif").toString());image.close();
        imagej.command().run(ch.epfl.biop.atlas.aligner.command.ImportSlicesFromFilesCommand.class,true,
                "mp",mp,"datasetname","synthetic","files",new java.io.File[]{output.resolve("synthetic.tif").toFile()},
                "split_rgb_channels",false,"slice_axis_initial_mm",5.0,"increment_between_slices_mm",.1).get();
        mp.waitForTasks();run(mp,output.toString());
        MultiSlicePositioner reopened=(MultiSlicePositioner)imagej.command().run(ch.epfl.biop.atlas.aligner.command.ABBAStartCommand.class,true,
                "x_axis","RL","y_axis","SI","z_axis","AP","ba",atlas).get().getOutput("mp");
        require(reopened.loadState(output.resolve("smoke.abba").toFile()),"Saved project loads");
        reopened.waitForTasks();verifyReload(mp,reopened);
        imagej.context().dispose();
        System.out.println("Synthetic run files: "+output);
        System.exit(0);
    }
    private static void require(boolean condition, String message) {
        if (!condition) throw new AssertionError(message);
    }
    private static AffineTransform3D world(SliceSources slice) {
        AffineTransform3D matrix = new AffineTransform3D();
        slice.getRegisteredSources()[0].getSpimSource().getSourceTransform(0,0,matrix);
        return matrix;
    }
    private static JsonArray update(double translation, Double position) {
        JsonObject value = new JsonObject(); value.addProperty("id","section_0001.tif");
        value.add("affine_mm",JsonParser.parseString("[[1,0,0,"+translation+"],[0,1,0,0],[0,0,1,0]]"));
        if (position != null) value.addProperty("position_mm",position);
        JsonArray updates = new JsonArray(); updates.add(value); return updates;
    }
    private static void translated(SliceSources slice, AffineTransform3D baseline, double delta) { translated(slice,baseline,delta,0); }
    /** delta: in-plane x shift; dz: the slicing-axis move that a position update adds. */
    private static void translated(SliceSources slice, AffineTransform3D baseline, double delta, double dz) {
        AffineTransform3D actual = world(slice);
        for(int r=0;r<3;r++) for(int c=0;c<4;c++) {
            double expected=baseline.get(r,c)+(r==0 && c==3?delta:0)+(r==2 && c==3?dz:0);
            require(Math.abs(actual.get(r,c)-expected)<1e-8,"Native affine entry "+r+","+c+" was "+actual.get(r,c)+" expected "+expected);
        }
    }
    public static void run(MultiSlicePositioner mp, String output) throws Exception {
        require(mp.getSlices().size()==1,"Supply exactly one synthetic section");
        SliceSources slice=mp.getSlices().get(0);
        AffineTransform3D original=world(slice); double initialZ=slice.getSlicingAxisPosition();
        int baseline=slice.getNumberOfRegistrations();
        AbbaHostSession host=new AbbaHostSession(mp,Paths.get(output,"snapshots"));
        JsonObject spec=JsonParser.parseString("{\"tasks\":[\"reorder\",\"position\",\"transform\"]}").getAsJsonObject();
        java.util.Map<SliceSources,String> damaged=new java.util.HashMap<>();damaged.put(slice,"torn");
        JsonObject request=host.prepare(spec,mp.getSlices(),java.util.Arrays.asList(0),25.0,damaged,true);
        require(slice.getNumberOfRegistrations()==baseline,"Calibration leaves no registration");
        require(Math.abs(slice.getSlicingAxisPosition()-initialZ)<1e-8,"Calibration restores position");
        translated(slice,original,0);
        require(Files.isRegularFile(Paths.get(output,"snapshots","section_0001.tif")),"Snapshot exported");
        require(request.getAsJsonArray("locked").size()==0,"Unregistered slices are never locked");
        require(request.getAsJsonObject("damaged").get("section_0001.tif").getAsString().equals("torn"),"User damage check reaches the request");
        AbbaHostSession pages=new AbbaHostSession(mp,Paths.get(output,"pages"));
        pages.prepare(spec,mp.getSlices(),java.util.Arrays.asList(0,0),25.0,new java.util.HashMap<>(),true);
        ij.ImagePlus multi=new ij.io.Opener().openImage(Paths.get(output,"pages","section_0001.tif").toString());
        require(multi.getStackSize()==2,"One page per exported channel");
        java.awt.image.BufferedImage shown=pages.exportPreview(slice,java.util.Arrays.asList(0),25.0,Paths.get(output,"preview.tif"));
        require(shown.getWidth()==multi.getWidth() && Files.isRegularFile(Paths.get(output,"preview.tif")),"Preview snapshot shares the run framing");
        double ap=request.getAsJsonObject("positions_mm").get("section_0001.tif").getAsDouble();
        require(Double.isFinite(ap) && ap>0,"Measured AP coordinate finite");
        // One final application per run: a single undoable batch replaces nothing and appends one registration.
        host.apply(update(.4,ap+.2)); mp.waitForTasks();
        require(slice.getNumberOfRegistrations()==baseline+1,"Final updates append one native affine");
        translated(slice,original,.4,slice.getSlicingAxisPosition()-initialZ);
        require(Math.abs(slice.getSlicingAxisPosition()-initialZ-.2)<1e-5,"Calibrated AP update reaches ABBA");
        mp.cancelLastAction(); mp.waitForTasks();
        require(slice.getNumberOfRegistrations()==baseline,"One Undo reverts the whole run");
        require(Math.abs(slice.getSlicingAxisPosition()-initialZ)<1e-5,"Undo restores the position");
        translated(slice,original,0);
        mp.redoAction(); mp.waitForTasks(); translated(slice,original,.4,slice.getSlicingAxisPosition()-initialZ);
        require(Math.abs(slice.getSlicingAxisPosition()-initialZ-.2)<1e-5,"Redo restores the position");
        require(slice.getNumberOfRegistrations()==baseline+1,"Redo restores the run");
        AbbaHostSession locked=new AbbaHostSession(mp,Paths.get(output,"locked"));
        require(locked.prepare(spec,mp.getSlices(),java.util.Arrays.asList(0),25.0,new java.util.HashMap<>(),true)
                .getAsJsonArray("locked").get(0).getAsString().equals("section_0001.tif"),"Registered slices are locked when overwriting is off");
        require(locked.prepare(spec,mp.getSlices(),java.util.Arrays.asList(0),25.0,new java.util.HashMap<>(),false)
                .getAsJsonArray("locked").size()==0,"Overwrite on sends no locks");
        // Registration changed outside the run: the stale session refuses to apply.
        try { pages.apply(update(.1,null)); throw new AssertionError("Expected refusal"); }
        catch (IllegalStateException expected) { require(expected.getMessage().contains("outside this run"),"Outside change is refused"); }
        mp.waitForTasks();
        require(mp.saveState(Paths.get(output,"smoke.abba").toFile(),true),"Native ABBA state saves");
        System.out.println("HOST_SESSION_PASS calibrated AP="+ap+", locked/damaged, multi-page, one-step apply/undo/redo, outside-change refusal, native save");
    }
    public static void verifyReload(MultiSlicePositioner before, MultiSlicePositioner after) {
        require(after.getSlices().size()==1,"Saved session reload retains section");
        SliceSources original=before.getSlices().get(0), restored=after.getSlices().get(0);
        require(restored.getNumberOfRegistrations()==original.getNumberOfRegistrations(),"Native registration reload retains stack");
        require(Math.abs(restored.getSlicingAxisPosition()-original.getSlicingAxisPosition())<1e-8,"Native reload retains AP position");
        AffineTransform3D a=world(original), b=world(restored);
        for(int r=0;r<3;r++) for(int c=0;c<4;c++)
            require(Math.abs(a.get(r,c)-b.get(r,c))<1e-8,"Native reload retains calibrated transform");
        System.out.println("HOST_RELOAD_PASS native project reopened with registration and position intact");
    }

}
