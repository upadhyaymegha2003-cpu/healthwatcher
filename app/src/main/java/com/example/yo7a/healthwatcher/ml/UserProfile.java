package com.example.yo7a.healthwatcher.ml;

import java.io.Serializable;

/**
 * Demographic inputs stored locally by the app (see {@code UserDB}). They are used by the original
 * blood-pressure formula and, only when a model manifest explicitly lists "demo_*" features, by ML models.
 * {@code gender}: 1 = male, 2 = female (the encoding already used by Login/UserDB).
 */
public final class UserProfile implements Serializable {
    private static final long serialVersionUID = 1L;

    public final int age;
    public final int heightCm;
    public final int weightKg;
    public final int gender;

    public UserProfile(int age, int heightCm, int weightKg, int gender) {
        this.age = age;
        this.heightCm = heightCm;
        this.weightKg = weightKg;
        this.gender = gender;
    }

    /** True when all values are in a physiologically sensible range. */
    public boolean isPlausible() {
        return age >= 1 && age <= 120 && heightCm >= 50 && heightCm <= 250
                && weightKg >= 10 && weightKg <= 350 && (gender == 1 || gender == 2);
    }
}
