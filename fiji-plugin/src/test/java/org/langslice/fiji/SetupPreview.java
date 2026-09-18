package org.langslice.fiji;

import java.awt.image.BufferedImage;
import java.io.File;
import java.lang.reflect.Constructor;
import javax.imageio.ImageIO;
import javax.swing.SwingUtilities;

/** Optional visual QA, run with a display: SetupPreview /path/to/preview.png. */
public final class SetupPreview {
    public static void main(String[] args) throws Exception {
        SwingUtilities.invokeAndWait(() -> {
            try {
                Constructor<SetupDialog> constructor = SetupDialog.class.getDeclaredConstructor(Runnable.class);
                constructor.setAccessible(true);
                SetupDialog dialog = constructor.newInstance((Runnable) null);
                dialog.setVisible(true);
                BufferedImage image = new BufferedImage(dialog.getWidth(), dialog.getHeight(), BufferedImage.TYPE_INT_RGB);
                java.awt.Graphics2D graphics = image.createGraphics();
                dialog.paint(graphics); graphics.dispose();
                ImageIO.write(image, "png", new File(args[0])); dialog.dispose();
            } catch (Exception failure) { throw new IllegalStateException(failure); }
        });
    }
}
