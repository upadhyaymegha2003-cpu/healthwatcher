package com.example.yo7a.healthwatcher;

import android.content.ContentValues;
import android.content.Context;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;
import android.database.sqlite.SQLiteOpenHelper;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by Yo7A on 4/17/2017.
 */

class user {
    private String username;
    private String name;
    private String password;
    private String email;
    private int age;
    private int height;
    private int weight;
    private int gender;

    public String getUsername() {
        return username;
    }

    public String getemail() {
        return email;
    }

    public String getname() {
        return name;
    }

    public String getPass() {
        return password;
    }

    public int getage() {
        return age;
    }

    public int getheight() {
        return height;
    }

    public int getweight() {
        return weight;
    }

    public int getgender() {
        return gender;
    }


    public void setUsername(String usrname) {

        username = usrname;
    }

    public void setemail(String E) {

        email = E;
    }

    public void setname(String nam) {

        name = nam;
    }

    public void setPass(String pass) {

        password = pass;
    }

    public void setage(int g) {

        age = g;
    }

    public void setheight(int h) {

        height = h;
    }

    public void setweight(int w) {

        weight = w;
    }

    public void setgender(int gen) {

        gender = gen;
    }
}

public class UserDB extends SQLiteOpenHelper {
    private static final int DATABASE_VERSION = 2;
    private static final String DATABASE_NAME = "userDB.db";

    //columns of the user Info table
    private static final String TABLE = "users";
    private static final String USERNAME = "username";
    private static final String NAME = "name";
    private static final String PASSWORD = "password";
    private static final String AGE = "age";
    private static final String HEIGHT = "height";
    private static final String WEIGHT = "weight";
    private static final String GENDER = "gender";
    private static final String EMAIL = "email";

    SQLiteDatabase db;

    public UserDB(Context context) {
        super(context, DATABASE_NAME, null, DATABASE_VERSION);
    }

    /** Opens a separate database file (used by instrumented tests so real user data is never touched). */
    public UserDB(Context context, String databaseName) {
        super(context, databaseName, null, DATABASE_VERSION);
    }


    //checks if the password is correct
    public String checkPass(String user) {
        db = this.getReadableDatabase();
        String query = "select username, password from users";
        Cursor cursor = db.rawQuery(query, null);
        String a, b;
        b = "Not found";

        if (cursor.moveToFirst()) {
            do {
                a = cursor.getString(0);

                if (a.equals(user)) {
                    b = cursor.getString(1);
                    break;
                }

            } while (cursor.moveToNext());
        }
        cursor.close();
        return b;
    }

    //checks if the username exist if yes it will return 0 otherwise it will return 1
    public int checkUser(String user) {
        db = this.getReadableDatabase();
        String query = "select username from users";
        Cursor cursor = db.rawQuery(query, null);
        String a;
        int x = 1;

        if (cursor.moveToFirst()) {
            do {
                a = cursor.getString(0);

                if (a.equals(user)) {
                    x = 0;
                    break;
                }
            } while (cursor.moveToNext());
        }
        cursor.close();
        return x;
    }

    public String getweight(String user) {
        db = this.getReadableDatabase();
        String query = "select username, weight from users";
        Cursor cursor = db.rawQuery(query, null);
        String a, w;
        w = "80";
        if (cursor.moveToFirst()) {
            do {
                a = cursor.getString(0);

                if (a.equals(user)) {
                    w = cursor.getString(1);
                    break;
                }

            } while (cursor.moveToNext());
        }
        cursor.close();
        return w;
    }

    public String getheight(String user) {
        db = this.getReadableDatabase();
        String query = "select username, height from users";
        Cursor cursor = db.rawQuery(query, null);
        String a, h;
        h = "180";

        if (cursor.moveToFirst()) {
            do {
                a = cursor.getString(0);

                if (a.equals(user)) {
                    h = cursor.getString(1);
                    break;
                }

            } while (cursor.moveToNext());
        }
        cursor.close();
        return h;
    }

    public String getage(String user) {
        db = this.getReadableDatabase();
        String query = "select username, age from users";
        Cursor cursor = db.rawQuery(query, null);
        String a, ag;
        ag = "26";

        if (cursor.moveToFirst()) {
            do {
                a = cursor.getString(0);

                if (a.equals(user)) {
                    ag = cursor.getString(1);
                    break;
                }

            } while (cursor.moveToNext());
        }
        cursor.close();
        return ag;
    }

    public String getgender(String user) {
        db = this.getReadableDatabase();
        String query = "select username, gender from users";
        Cursor cursor = db.rawQuery(query, null);
        String a, g;
        g = "2";

        if (cursor.moveToFirst()) {
            do {
                a = cursor.getString(0);

                if (a.equals(user)) {
                    g = cursor.getString(1);
                    break;
                }

            } while (cursor.moveToNext());
        }
        cursor.close();
        return g;
    }

    public void addUser(user u) {
        db = this.getWritableDatabase();
        ContentValues values = new ContentValues();
        values.put(USERNAME, u.getUsername());
        values.put(NAME, u.getname());
        values.put(PASSWORD, u.getPass());
        values.put(AGE, u.getage());
        values.put(HEIGHT, u.getheight());
        values.put(EMAIL, u.getemail());
        values.put(WEIGHT, u.getweight());
        values.put(GENDER, u.getgender());
        db.insert(TABLE, null, values);
        db.close();
    }

    // ---------------------------------------------------------------- measurement history (schema v2)

    private static final String CREATE_MEASUREMENTS =
            "create table if not exists measurements ( id integer primary key autoincrement, "
                    + "username text not null, ts integer not null, hr real, sbp real, dbp real, rr real, spo2 real, "
                    + "sources text, model_versions text, quality_status text, quality_note text, "
                    + "duration_s real, sampling_hz real);";
    private static final String CREATE_MEASUREMENTS_INDEX =
            "create index if not exists idx_measurements_user_ts on measurements (username, ts);";

    private static Double nullable(double v) {
        return Double.isNaN(v) ? null : v;
    }

    private static double read(Cursor c, int col) {
        return c.isNull(col) ? Double.NaN : c.getDouble(col);
    }

    /** Stores a measurement; returns the new row id or -1 on failure. */
    public long addMeasurement(MeasurementRecord m) {
        SQLiteDatabase w = this.getWritableDatabase();
        ContentValues v = new ContentValues();
        v.put("username", m.username);
        v.put("ts", m.timestampMillis);
        v.put("hr", nullable(m.heartRate));
        v.put("sbp", nullable(m.systolic));
        v.put("dbp", nullable(m.diastolic));
        v.put("rr", nullable(m.respiratoryRate));
        v.put("spo2", nullable(m.spo2));
        v.put("sources", m.sources);
        v.put("model_versions", m.modelVersions);
        v.put("quality_status", m.qualityStatus);
        v.put("quality_note", m.qualityNote);
        v.put("duration_s", m.durationSec);
        v.put("sampling_hz", m.samplingHz);
        return w.insert("measurements", null, v);
    }

    /** Newest first. */
    public List<MeasurementRecord> getMeasurements(String user, int limit) {
        List<MeasurementRecord> out = new ArrayList<>();
        SQLiteDatabase r = this.getReadableDatabase();
        Cursor c = r.rawQuery("select id, username, ts, hr, sbp, dbp, rr, spo2, sources, model_versions, "
                + "quality_status, quality_note, duration_s, sampling_hz from measurements "
                + "where username = ? order by ts desc limit ?", new String[]{user, String.valueOf(limit)});
        try {
            while (c.moveToNext()) {
                MeasurementRecord m = new MeasurementRecord();
                m.id = c.getLong(0);
                m.username = c.getString(1);
                m.timestampMillis = c.getLong(2);
                m.heartRate = read(c, 3);
                m.systolic = read(c, 4);
                m.diastolic = read(c, 5);
                m.respiratoryRate = read(c, 6);
                m.spo2 = read(c, 7);
                m.sources = c.isNull(8) ? "" : c.getString(8);
                m.modelVersions = c.isNull(9) ? "" : c.getString(9);
                m.qualityStatus = c.isNull(10) ? "" : c.getString(10);
                m.qualityNote = c.isNull(11) ? "" : c.getString(11);
                m.durationSec = read(c, 12);
                m.samplingHz = read(c, 13);
                out.add(m);
            }
        } finally {
            c.close();
        }
        return out;
    }

    /** Latest stored measurement of the user, or null. */
    public MeasurementRecord getLatestMeasurement(String user) {
        List<MeasurementRecord> l = getMeasurements(user, 1);
        return l.isEmpty() ? null : l.get(0);
    }

    public int deleteMeasurements(String user) {
        return this.getWritableDatabase().delete("measurements", "username = ?", new String[]{user});
    }

    @Override
    public void onCreate(SQLiteDatabase db) {
        String createTable = "create table users ( username text primary key not null, " +
                " name text not null, password text not null, email text not null, age INTEGER not null, height INTEGER not null, weight INTEGER not null, gender INTEGER not null);";
        db.execSQL(createTable);
        db.execSQL(CREATE_MEASUREMENTS);
        db.execSQL(CREATE_MEASUREMENTS_INDEX);
        this.db = db;
    }

    /** Upgrades keep existing user accounts: v1 -> v2 only adds the measurements table. */
    @Override
    public void onUpgrade(SQLiteDatabase db, int oldVersion, int newVersion) {
        if (oldVersion < 2) {
            db.execSQL(CREATE_MEASUREMENTS);
            db.execSQL(CREATE_MEASUREMENTS_INDEX);
        }
        this.db = db;
    }
}
