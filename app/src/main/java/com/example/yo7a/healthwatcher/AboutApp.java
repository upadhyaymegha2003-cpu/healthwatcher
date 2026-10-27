package com.example.yo7a.healthwatcher;

import android.content.Intent;
import android.os.Bundle;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/** About screen, including the live status of the on-device ML models. */
public class AboutApp extends AppCompatActivity {
    private String user = "";
    private ExecutorService io;
    private volatile boolean destroyed = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_about_app);
        Bundle bundle = getIntent().getExtras();
        if (bundle != null && bundle.getString(MeasurementActivity.EXTRA_USER) != null) {
            user = bundle.getString(MeasurementActivity.EXTRA_USER);
        }
        final TextView status = findViewById(R.id.modelStatus);
        status.setText("Checking installed models\u2026");
        io = Executors.newSingleThreadExecutor();
        io.execute(new Runnable() {
            @Override
            public void run() {
                final String text = ModelStatus.describe(AboutApp.this);
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        if (!destroyed) {
                            status.setText(text);
                        }
                    }
                });
            }
        });
    }

    @Override
    protected void onDestroy() {
        destroyed = true;
        io.shutdownNow();
        super.onDestroy();
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
