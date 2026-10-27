package com.example.yo7a.healthwatcher;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import android.content.Context;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;

import androidx.test.platform.app.InstrumentationRegistry;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.File;
import java.util.List;

/** Instrumented tests for the history table and the non-destructive v1 -> v2 migration. */
public class MeasurementDbInstrumentedTest {
    private static final String USER = "unit_test_user";
    private static final String TEST_DB = "userDB_instrumented_test.db";
    private Context ctx;

    @Before
    public void setUp() {
        ctx = InstrumentationRegistry.getInstrumentation().getTargetContext();
    }

    @After
    public void tearDown() {
        ctx.deleteDatabase(TEST_DB);
    }

    private static MeasurementRecord record(long ts, double hr, double sbp, double dbp) {
        MeasurementRecord m = new MeasurementRecord();
        m.username = USER;
        m.timestampMillis = ts;
        m.heartRate = hr;
        m.systolic = sbp;
        m.diastolic = dbp;
        m.sources = "heart_rate=SIGNAL_PROCESSING";
        m.modelVersions = "";
        m.qualityStatus = "OK";
        m.durationSec = 30;
        m.samplingHz = 29.5;
        return m;
    }

    @Test
    public void storesAndReadsHistoryNewestFirstWithNulls() {
        UserDB db = new UserDB(ctx, TEST_DB);
        assertTrue(db.addMeasurement(record(1000, 70, Double.NaN, Double.NaN)) > 0);
        assertTrue(db.addMeasurement(record(3000, 80, 118, 76)) > 0);
        assertTrue(db.addMeasurement(record(2000, 75, Double.NaN, Double.NaN)) > 0);

        List<MeasurementRecord> l = db.getMeasurements(USER, 10);
        assertEquals(3, l.size());
        assertEquals(3000, l.get(0).timestampMillis);
        assertEquals(1000, l.get(2).timestampMillis);
        assertEquals(118, l.get(0).systolic, 0);
        assertTrue("missing vitals stay NaN, not 0", Double.isNaN(l.get(1).systolic));
        assertEquals("SP", l.get(0).sourceTag(com.example.yo7a.healthwatcher.ml.VitalTarget.HEART_RATE));
        assertEquals(80, db.getLatestMeasurement(USER).heartRate, 0);
        assertEquals(1, db.getMeasurements(USER, 1).size());
    }

    @Test
    public void historyIsPerUserAndDeletable() {
        UserDB db = new UserDB(ctx, TEST_DB);
        db.addMeasurement(record(1, 60, Double.NaN, Double.NaN));
        assertTrue(db.getMeasurements("someone_else", 10).isEmpty());
        assertEquals(1, db.deleteMeasurements(USER));
        assertNull(db.getLatestMeasurement(USER));
    }

    @Test
    public void upgradeFromVersion1KeepsUsers() throws Exception {
        ctx.deleteDatabase(TEST_DB);
        File f = ctx.getDatabasePath(TEST_DB);
        f.getParentFile().mkdirs();
        SQLiteDatabase old = SQLiteDatabase.openOrCreateDatabase(f, null);
        old.execSQL("create table users ( username text primary key not null, name text not null, password text not null,"
                + " email text not null, age INTEGER not null, height INTEGER not null, weight INTEGER not null, gender INTEGER not null);");
        old.execSQL("insert into users values ('legacy','Legacy User','pw','a@b.c',40,170,70,1)");
        old.setVersion(1);
        old.close();

        UserDB db = new UserDB(ctx, TEST_DB);                 // triggers onUpgrade(1 -> 2)
        SQLiteDatabase r = db.getReadableDatabase();
        Cursor c = r.rawQuery("select username, age from users where username = 'legacy'", null);
        try {
            assertTrue("existing account must survive the upgrade", c.moveToFirst());
            assertEquals(40, c.getInt(1));
        } finally {
            c.close();
        }
        assertNotNull(db.getMeasurements("legacy", 5));
        assertTrue(db.addMeasurement(record(5, 66, Double.NaN, Double.NaN)) > 0);
        db.deleteMeasurements(USER);
        ctx.deleteDatabase(TEST_DB);
    }
}
