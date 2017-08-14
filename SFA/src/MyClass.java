import sfa.timeseries.TimeSeries;
import sfa.timeseries.TimeSeriesLoader;
import sfa.transformation.SFA;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import static easyjcckit.QuickPlot.*;

/**
 * Created by student on 24/7/2017.
 */
public class MyClass {
    public static void main(String[] args) {
        System.out.println("Hello World!"); // Display the string.

        try {
            TimeSeries[] train = TimeSeriesLoader.loadDatset(new File("./datasets/CBF/CBF_TRAIN"));

            //TimeSeries[] small = new TimeSeries[]{};
            ArrayList<TimeSeries> small = new ArrayList<>();

            int count = 0;
            for(TimeSeries tt : train) {
                //System.out.println(tt.getLength());
                small.add(tt);
                count += 1;
                if(count > 10) {
                    break;
                }
            }

            //TimeSeries[] small = (TimeSeries[]) (Arrays.stream(train).limit(10).toArray());
            TimeSeries[] smalls = Arrays.copyOf(small.toArray(), small.size(), TimeSeries[].class);

            System.out.println(train.length);
            System.out.println(smalls.length);

            //return;
            sfait(train);
            System.out.println();
            sfait(smalls);

            /*double[] data = train[1].getData();

            *//*for(double dd : data) {
                System.out.println(dd);
            }*//*

            //System.out.println(data.length);
            //IntStream.range(0, 10);
            double[] xaxis = IntStream.range(0, data.length).mapToDouble(n -> n).toArray();

            //double[] yvalues = new double[]{0,1,4,9,16,25};
            double[] yvalues = data;

            plot( xaxis, yvalues ); // create a plot using xaxis and yvalues

            */

        } catch (IOException e) {
            e.printStackTrace();
        }

//        IntStream.range(0, 10).forEach(
//                n -> {
//                    System.out.println(n);
//                }
//        );

        //double[] yvalues = new double[]{0,1,2,3,4,5};
        //addPlot( xaxis, yvalues ); // create a second plot on top of first

        System.out.println("Press enter to exit");
        //System.in.read();
    }

    public static void sfait(TimeSeries[] ts) {
        SFA sfa = new SFA(SFA.HistogramType.EQUI_DEPTH);

        short[][] wordsTrain = sfa.fitTransform(ts, 10, 4, false);
        for(short[] word : wordsTrain) {
            String str = java.util.Arrays.toString(word);
            System.out.println(str);
        }
    }
}
