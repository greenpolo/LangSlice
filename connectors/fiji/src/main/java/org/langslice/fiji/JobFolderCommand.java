package org.langslice.fiji;
import ch.epfl.biop.atlas.aligner.MultiSlicePositioner;
import org.scijava.ItemIO;
import org.scijava.command.Command;
import org.scijava.plugin.*;
/**
 * The folder to open with LangSlice's start_job for this ABBA project (its section snapshots; the job is in its
 * langslice/ folder), for an assistant driving ABBA (abba-mcp) after Register › LangSlice… prepared the job. Empty while
 * there is none.
 */
@Plugin(type = Command.class, menuPath = "Plugins>LangSlice>LangSlice Job Folder",
        description = "The folder to pass to LangSlice's start_job for this ABBA project. Run Register > LangSlice... "
                + "(org.langslice.fiji.OpenLangSliceCommand) first: it prepares the job in the background, and ABBA then "
                + "follows every change the agent saves. 'ready' is true once the job exists.")
public final class JobFolderCommand implements Command {
    @Parameter private MultiSlicePositioner mp;
    @Parameter(type = ItemIO.OUTPUT, label = "Job folder") private String job_folder;
    @Parameter(type = ItemIO.OUTPUT, label = "Ready") private Boolean ready;
    @Override public void run() {
        ProjectJob.Found found = ProjectJob.find(mp);
        ready = found.folder != null && ProjectJob.hasJob(found.folder);
        job_folder = found.folder == null ? "" : found.folder.toString();
    }
}
