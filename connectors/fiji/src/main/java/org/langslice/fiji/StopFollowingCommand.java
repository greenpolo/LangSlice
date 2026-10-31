package org.langslice.fiji;
import ch.epfl.biop.atlas.aligner.MultiSlicePositioner;
import org.scijava.command.Command;
import org.scijava.plugin.*;
/** Detaches the follower from this ABBA session without closing ABBA; ABBA keeps what it has. */
@Plugin(type = Command.class, name = LangSliceService.STOP, menuPath = "Plugins>LangSlice>Stop following")
public final class StopFollowingCommand implements Command {
    @Parameter private MultiSlicePositioner mp;
    @Override public void run() {
        FijiLog.line(Follower.stop(mp) ? "LangSlice stopped following this ABBA session; ABBA keeps what it has. Choose Register > LangSlice… to follow again."
                : "LangSlice is not following this ABBA session.");
    }
}
