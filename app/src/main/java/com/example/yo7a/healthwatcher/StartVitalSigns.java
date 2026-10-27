package com.example.yo7a.healthwatcher;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

/** Pre-measurement instructions screen; starts the measurement activity matching the chosen page. */
public class StartVitalSigns extends AppCompatActivity {
    private String user = "";
    private MeasurementMode mode = MeasurementMode.ALL;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_start_vital_signs);
        Bundle extras = getIntent().getExtras();
        if (extras != null) {
            String u = extras.getString(MeasurementActivity.EXTRA_USER);
            user = u == null ? "" : u;
            mode = MeasurementMode.fromPage(extras.getInt("Page", MeasurementMode.ALL.page));
        }
        ((TextView) findViewById(R.id.startTitle)).setText(titleFor(mode));
        findViewById(R.id.StartVS).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent i = new Intent(StartVitalSigns.this, activityFor(mode));
                i.putExtra(MeasurementActivity.EXTRA_USER, user);
                startActivity(i);
                finish();
            }
        });
    }

    private static Class<?> activityFor(MeasurementMode m) {
        switch (m) {
            case HEART_RATE:
                return HeartRateProcess.class;
            case BLOOD_PRESSURE:
                return BloodPressureProcess.class;
            case RESPIRATION:
                return RespirationProcess.class;
            case SPO2:
                return O2Process.class;
            default:
                return VitalSignsProcess.class;
        }
    }

    private static String titleFor(MeasurementMode m) {
        switch (m) {
            case HEART_RATE:
                return "Measure heart rate";
            case BLOOD_PRESSURE:
                return "Estimate blood pressure";
            case RESPIRATION:
                return "Measure respiratory rate";
            case SPO2:
                return "Estimate oxygen saturation";
            default:
                return "Measure all vital signs";
        }
    }

    @Override
    public void onBackPressed() {
        Intent i = new Intent(this, Primary.class);
        i.putExtra(MeasurementActivity.EXTRA_USER, user);
        i.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP);
        startActivity(i);
        finish();
    }
}
