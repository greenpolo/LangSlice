package org.langslice.fiji;
import ch.epfl.biop.atlas.aligner.MultiSlicePositioner;
import org.scijava.ItemIO;
import org.scijava.command.Command;
import org.scijava.plugin.*;
/**
 * The LangSlice job folder of this ABBA project, for an assistant driving ABBA (abba-mcp): after Register › LangSlice…
 * prepared the job, the assistant opens this folder with LangSlice's start_job. Empty while there is no job yet.
 */
@Plugin(type = Command.class, menuPath = "Plugins>LangSlice>LangSlice Job Folder",
        description = "The LangSlice job folder of this ABBA project, for LangSlice's start_job. Run Register > LangSlice... "
                + "(org.langslice.fiji.OpenLangSliceCommand) first: it prepares the job in the background, and ABBA then "
                + "follows every change the agent saves. 'ready' is true once the job exists.")
public final class JobFolderCommand implements Command {
    @Parameter private MultiSlicePositioner mp;
    @Parameter(type = ItemIO.OUTPUT, label = "Job folder") private String job_folder;
    @Parameter(type = ItemIO.OUTPUT, label = "Ready") private Boolean ready;
    @Override public void run() {
        ProjectJob.Found found = ProjectJob.find(mp);
        ready = found.folder != null && ProjectJob.hasJob(found.folder);
        job_folder = found.folder == null ? "" : ProjectJob.jobFolder(found.folder).toString();
    }
}
