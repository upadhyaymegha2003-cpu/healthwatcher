import com.example.yo7a.healthwatcher.ml.BaselineEstimator;
import com.example.yo7a.healthwatcher.ml.PpgSignal;
import com.example.yo7a.healthwatcher.ml.QualityReport;
import com.example.yo7a.healthwatcher.ml.SignalQualityAnalyzer;
import com.example.yo7a.healthwatcher.ml.UserProfile;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;

/**
 * Runs the app's ORIGINAL signal-processing estimators and the quality gate on a dataset
 * (recordings.csv + signal files), so train.py can compare the trained model against them.
 * Usage: java BaselineCli DATASET_DIR OUT.csv
 */
public class BaselineCli {
    public static void main(String[] args) throws Exception {
        String dir = args[0];
        try (BufferedReader br = new BufferedReader(new FileReader(dir + "/recordings.csv"));
             PrintWriter out = new PrintWriter(args[1])) {
            String[] head = br.readLine().split(",", -1);
            out.println("recording_id,hr_bpm,rr_bpm,spo2,sbp,dbp,quality_status");
            String line;
            while ((line = br.readLine()) != null) {
                String[] p = line.split(",", -1);
                String id = get(head, p, "recording_id");
                try {
                    PpgSignal s = read(dir + "/" + get(head, p, "file"));
                    UserProfile prof = new UserProfile(i(get(head, p, "age")), i(get(head, p, "height_cm")),
                            i(get(head, p, "weight_kg")), i(get(head, p, "gender")));
                    BaselineEstimator.Result r = new BaselineEstimator().estimate(s, prof);
                    QualityReport q = new SignalQualityAnalyzer().analyze(s);
                    out.println(id + "," + n(r.heartRate) + "," + n(r.respiratoryRate) + "," + n(r.spo2) + ","
                            + n(r.systolic) + "," + n(r.diastolic) + "," + q.statusLabel());
                } catch (Exception e) {
                    out.println(id + ",,,,,,ERROR");
                }
            }
        }
    }

    static String get(String[] head, String[] p, String col) {
        for (int i = 0; i < head.length; i++) {
            if (head[i].equals(col)) return i < p.length ? p[i] : "";
        }
        return "";
    }

    static int i(String s) {
        try {
            return (int) Math.round(Double.parseDouble(s));
        } catch (Exception e) {
            return 0;
        }
    }

    static String n(double v) {
        return Double.isNaN(v) ? "" : Double.toString(v);
    }

    static PpgSignal read(String path) throws Exception {
        List<double[]> rows = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            br.readLine();
            String l;
            while ((l = br.readLine()) != null) {
                String[] p = l.split(",");
                rows.add(new double[]{Double.parseDouble(p[0]), Double.parseDouble(p[1]), Double.parseDouble(p[2]), Double.parseDouble(p[3])});
            }
        }
        int n = rows.size();
        double[] t = new double[n], r = new double[n], g = new double[n], b = new double[n];
        for (int k = 0; k < n; k++) {
            t[k] = rows.get(k)[0] - rows.get(0)[0];
            r[k] = rows.get(k)[1];
            g[k] = rows.get(k)[2];
            b[k] = rows.get(k)[3];
        }
        return new PpgSignal(t, r, g, b);
    }
}
