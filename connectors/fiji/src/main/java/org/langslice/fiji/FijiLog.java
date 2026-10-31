package org.langslice.fiji;

/** Fiji's Log window and status bar: where the connector says what it does (headless Fiji prints them). */
final class FijiLog {
    private FijiLog() { }

    static void line(String text) { ij.IJ.log(text); }

    static void status(String text) { ij.IJ.showStatus(text); }
}
