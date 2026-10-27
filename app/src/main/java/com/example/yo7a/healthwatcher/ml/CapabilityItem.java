package com.example.yo7a.healthwatcher.ml;

/** One row of the "System check" screen: a real, measured runtime status (never a hard-coded "OK"). */
public final class CapabilityItem {
    public enum Status {PASS, INFO, WARN, FAIL}

    public final String title;
    public final Status status;
    public final String detail;

    public CapabilityItem(String title, Status status, String detail) {
        this.title = title;
        this.status = status;
        this.detail = detail;
    }
}
