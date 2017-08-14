package sfa.test;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Random;

import sfa.index.SFATrie;
import sfa.index.SortedListMap;
import sfa.timeseries.TimeSeries;
import sfa.timeseries.TimeSeriesLoader;

public class SFATrieTest {
  static int l = 20; // SFA word length ( & dimensionality of the index)
  static int leafThreshold = 10; // number of subsequences in each leaf node
  static int k = 1; // k-NN search

public static void testWholeMatching() throws IOException {
    int N = 100000;

    TimeSeries[] timeSeries2 = TimeSeriesLoader.readSamplesQuerySeries(new File("./datasets/indexing/query_lightcurves.txt"));
    int n = timeSeries2[0].getLength(); 
    System.out.println("Queries: " + timeSeries2.length);

    System.out.println("Generating Time Series");
    TimeSeries[] timeSeries = new TimeSeries[N];
    for (int i = 0; i < N; i++) {
      timeSeries[i] = TimeSeriesLoader.generateRandomWalkData(n, new Random(i));
      timeSeries[i].norm();
    }
    System.out.println("Whole Series: " + timeSeries.length);

    Runtime runtime = Runtime.getRuntime();
    long mem = runtime.totalMemory();
    long time = System.currentTimeMillis();
    
    SFATrie index = new SFATrie(l, leafThreshold);
    index.buildIndexWholeMatching(timeSeries);
    index.checkIndex();
    
    // GC
    performGC();
    System.out.println("Memory: " + ((runtime.totalMemory() - mem) / (1048576l)) + " MB (rough estimate)");

    System.out.println("Perform NN-queries");    
    for (int i = 0; i < timeSeries2.length; i++) {
      System.out.println((i+1) + ". Query");
      TimeSeries query = timeSeries2[i];
      
      time = System.currentTimeMillis();
      SortedListMap<Double, Integer> result = index.searchNearestNeighbor(query, k);
      time = System.currentTimeMillis() - time;
      System.out.println("\tSFATree:" + (time/1000.0) + "s");

      List<Double> distances = result.keys();

      System.out.println("\tTS seen: " +index.getTimeSeriesRead() + " " + index.getTimeSeriesRead()/(double)N + "%");
      System.out.println("\tLeaves seen " + index.getIoTimeSeriesRead());
      System.out.println("\tNodes seen " +  index.getBlockRead());

      index.resetIoCosts();

      // compare with 1-NN ED search
      time = System.currentTimeMillis();
      double resultDistance = Double.MAX_VALUE;
      for (int w = 0; w < N; w++) {
        double distance = getEuclideanDistance(timeSeries[w], query, 0.0, 1.0, resultDistance, 0);
        resultDistance = Math.min(distance, resultDistance);
      }
      time = System.currentTimeMillis() - time;
      System.out.println("\tEuclidean:" + (time/1000.0) + "s");

      if (distances.get(0) != resultDistance) {
        System.out.println("\tError! Distances do not match: " + resultDistance + "\t" + distances.get(0));
      }
      else {
        System.out.println("\tDistance is ok");
      }
    }

    System.out.println("All ok...");
  }

  public static void testSubsequenceMatching() throws IOException {
    
    System.out.println("Loading Time Series");
//    TimeSeries timeSeries = TimeSeriesLoader.readSampleSubsequence(new File("./datasets/indexing/sample_lightcurves.txt"));
    TimeSeries timeSeries = TimeSeriesLoader.generateRandomWalkData(100000, new Random(1));    
    System.out.println("Sample DS size : " + timeSeries.getLength());

    TimeSeries[] timeSeries2 = TimeSeriesLoader.readSamplesQuerySeries(new File("./datasets/indexing/query_lightcurves.txt"));
    int windowLength = timeSeries2[0].getLength(); // length of the subsequences to be indexed
    System.out.println("Query DS size : " + windowLength);

    Runtime runtime = Runtime.getRuntime();
    long mem = runtime.totalMemory();
    long time = System.currentTimeMillis();
    
    SFATrie index = new SFATrie(l, leafThreshold);
    index.buildIndexSubsequenceMatching(timeSeries, windowLength);
    index.checkIndex();
    
    // GC
    performGC();
    System.out.println("Memory: " + ((runtime.totalMemory() - mem) / (1048576l)) + " MB (rough estimate)");

    System.out.println("Perform NN-queries");
    int size = (timeSeries.getData().length-windowLength)+1;
    double[] means = new double[size];
    double[] stds = new double[size];
    TimeSeries.calcIncreamentalMeanStddev(windowLength, timeSeries.getData(), means, stds);
    
    for (int i = 0; i < timeSeries2.length; i++) {
      System.out.println((i+1) + ". Query");
      TimeSeries query = timeSeries2[i];
      
      time = System.currentTimeMillis();
      SortedListMap<Double, Integer> result = index.searchNearestNeighbor(query, k);
      time = System.currentTimeMillis() - time;
      System.out.println("\tSFATree:" + (time/1000.0) + "s");

      List<Double> distances = result.keys();

      System.out.println("\tTS seen: " +index.getTimeSeriesRead() + " " + index.getTimeSeriesRead()/(double)size + "%");
      System.out.println("\tLeaves seen " + index.getIoTimeSeriesRead());
      System.out.println("\tNodes seen " +  index.getBlockRead());

      index.resetIoCosts();

      // compare with 1-NN ED search
      time = System.currentTimeMillis();
      double resultDistance = Double.MAX_VALUE;
      for (int ww = 0; ww < size; ww++) {
        double distance = getEuclideanDistance(timeSeries, query, means[ww], stds[ww], resultDistance, ww);
        resultDistance = Math.min(distance, resultDistance);
      }
      time = System.currentTimeMillis() - time;
      System.out.println("\tEuclidean:" + (time/1000.0) + "s");

      if (distances.get(0) != resultDistance) {
        System.out.println("\tError! Distances do not match: " + resultDistance + "\t" + distances.get(0));
      }
      else {
        System.out.println("\tDistance is ok");
      }
    }

    System.out.println("All ok...");
  }

  public static void performGC() {
    try {
      System.gc();
      Thread.sleep(10);
    } catch (InterruptedException e) {
      e.printStackTrace();
    }
  }

  public static double getEuclideanDistance(
      TimeSeries ts,
      TimeSeries q,
      double meanTs,
      double stdTs,
      double minValue,
      int w
      ) {

    // 1 divided by stddev for fastert calculations
    stdTs = (stdTs>0? 1.0 / stdTs : 1.0);

    double distance = 0.0;
    double[] tsData = ts.getData();
    double[] qData = q.getData();

    for (int ww = 0; ww < qData.length; ww++) {
      double value1 = (tsData[w+ww]-meanTs) * stdTs;
      double value = qData[ww] - value1;
      distance += value*value;

      // early abandoning
      if (distance >= minValue) {
        return Double.MAX_VALUE;
      }
    }

    return distance;
  }

  public static void main(String argv[]) throws IOException {
    testWholeMatching();
    testSubsequenceMatching();
  }
}
