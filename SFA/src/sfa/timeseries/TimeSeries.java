// Copyright (c) 2016 - Patrick Schäfer (patrick.schaefer@zib.de)
// Distributed under the GLP 3.0 (See accompanying file LICENSE)
package sfa.timeseries;

import java.io.Serializable;
import java.util.Arrays;

public class TimeSeries implements Serializable {
  private static final long serialVersionUID = 6340030797230203868L;

  protected double[] data = null;

  protected double mean = 0;
  protected double stddev = 1;

  protected boolean normed = false;
  protected String label = null;

  public TimeSeries (double[] data) {
    this.data = data;
  }
  
  public TimeSeries (double[] data, String label) {
    this(data);
    this.label = label;
  }

  public boolean isNormed() {
    return this.normed;
  }

  public double[] getData() {
    return this.data;
  }

  public double getData(int i) {
    return this.data[i];
  }

  public void setData(double[] init) {
    this.data = init;
  }

  public int getLength() {
    return this.data==null? 0 : this.data.length;
  }

  public void norm() {
    norm(true);
  }

  public void norm (boolean norm) {
    this.mean = calculateMean();
    this.stddev = calculateStddev();

    if (!isNormed()) {
      norm(norm, this.mean, this.stddev);
    }
  }

  /**
   * Zero-mean normalization. After normalization the following holds:
   *    mean = 0
   *    stddev = 1
   */
  public void norm (boolean normMean, double mean, double stddev) {
    this.mean = mean;
    this.stddev = stddev;

    if (!isNormed()) {
      double inverseStddev = (this.stddev != 0)? 1.0 / this.stddev : 1.0;

      if (normMean) {
        for (int i = 0; i < this.data.length; i++) {
          this.data[i] = (this.data[i]-this.mean) *  inverseStddev;
        }
        this.mean = 0.0;
      }
      else if (inverseStddev != 1.0) {
        for (int i = 0; i < this.data.length; i++) {
          this.data[i] *= inverseStddev;
        }
      }

//      this.stddev = 1.0;
      this.normed = true;
    }
  }

  public double calculateStddev() {
    this.stddev=0;

    // stddev
    double var = 0;
    for (double value : getData()) {
      var += value * value;
    }

    double norm = 1.0 / ((double)this.data.length);
    double buf = norm * var - this.mean*this.mean;
    if (buf > 0) {
      this.stddev = Math.sqrt(buf);
    }

    return this.stddev;
  }


  public double calculateMean() {
    this.mean=0.0;

    // get mean values
    for (double value : getData()) {
      this.mean += value;
    }
    this.mean /= (double)this.data.length;

    return this.mean;
  }
  
  public double getStddev() {
    return this.stddev;
  }

  public double getMean() {
    return this.mean;
  }

  @Override
  public boolean equals(Object o) {
    if (!(o instanceof TimeSeries)) {
      throw new RuntimeException("Objects should be time series: " + (o.toString()));
    }

    TimeSeries ts = (TimeSeries) o;

    // Only compare normed time series
    if (ts.isNormed() != this.isNormed()) {
      throw new RuntimeException("Please norm both time series");
    }


    // same dimensionality?
    if (ts.getLength() != this.getLength()) {
      return false;
    }

    for (int i = 0; i < ts.getLength(); i++) {
      // search for differences
      if (ts.data[i] != this.data[i]) {
        return false;
      }
    }

    return true;
  }

  
  /**
   * Get a subsequence starting at offset with length windowSize.
   * @param windowSize
   * @return
   */
  public TimeSeries getSubsequence(int offset, int windowSize) {
    double[] subsequenceData = Arrays.copyOfRange(this.data, offset, Math.min(data.length, offset+windowSize));
    TimeSeries sequence = new TimeSeries(subsequenceData);
    sequence.norm();
    return sequence;
  }

  /**
   * Get a subsequence starting at offset with length windowSize.
   * @param windowSize
   * @return
   */
  public TimeSeries getSubsequence(int offset, int windowSize, double mean, double stddev) {
    double[] subsequenceData = Arrays.copyOfRange(this.data, offset, offset+windowSize);
    if (subsequenceData.length != windowSize) {
      System.err.println("Wrong size!!");
    }
    TimeSeries sequence = new TimeSeries(subsequenceData);
    sequence.norm(true, mean, stddev);
    return sequence;
  }
  
  /**
   * Get sliding windows with windowSize.
   * @param windowSize
   * @return
   */
  public TimeSeries[] getSubsequences(int windowSize, boolean normMean) {
    // Window offset
    int offset = 1;

    // extract subsequences
    int size = (this.data.length-windowSize)/offset+1;
    TimeSeries[] subsequences = new TimeSeries[size];

    double[] means = new double[size];
    double[] stddevs = new double[size];

    calcIncreamentalMeanStddev(windowSize, this.data, means, stddevs);

    int pos = 0;
    for (int i=0; i < subsequences.length; i++) {
      double subsequenceData[] = new double[windowSize];
      System.arraycopy(this.data, pos, subsequenceData, 0, windowSize);

      // Die neu erstellten Zeitreihen haben die L�nge windowSize und das Offset i
      subsequences[i] = new TimeSeries(subsequenceData);
      if (offset == 1) {
        subsequences[i].norm(normMean, means[i], stddevs[i]);
      }
      else {
        subsequences[i].norm(normMean);
      }
      subsequences[i].setLabel(getLabel());

      pos += offset;
    }
    return subsequences;
  }

  /**
   * Gets the means and stddevs for all sliding windows of a time series
   */
  public static void calcIncreamentalMeanStddev(
      int windowLength,
      double[] ts,
      double[] means,
      double[] stds) {
    double sum = 0;
    double squareSum = 0;

    // it is faster to multiply than to divide
    double rWindowLength = 1.0 / (double)windowLength;

    double[] tsData = ts;
    for (int ww = 0; ww < windowLength; ww++) {
      sum += tsData[ww];
      squareSum += tsData[ww]*tsData[ww];
    }
    means[0] = sum * rWindowLength;
    double buf = squareSum * rWindowLength - means[0]*means[0];
    stds[0] = buf > 0? Math.sqrt(buf) : 0;

    for (int w = 1, end = tsData.length-windowLength+1; w < end; w++) {
      sum += tsData[w+windowLength-1] - tsData[w-1];
      means[w] = sum * rWindowLength;

      squareSum += tsData[w+windowLength-1]*tsData[w+windowLength-1] - tsData[w-1]*tsData[w-1];
      buf = squareSum * rWindowLength - means[w]*means[w];
      stds[w] = buf > 0? Math.sqrt(buf) : 0;
    }
  }
  
  /**
   * Extracts disjoint subsequences
   * @param windowSize
   * @return
   */
  public TimeSeries[] getDisjointSequences(int windowSize, boolean normMean) {

    // extract subsequences
    int amount = getLength()/windowSize;
    TimeSeries[] subsequences = new TimeSeries[amount];

    for (int i=0; i < amount; i++) {
      double subsequenceData[] = new double[windowSize];
      System.arraycopy(this.data, i*windowSize, subsequenceData, 0, windowSize);
      subsequences[i] = new TimeSeries(subsequenceData);
      subsequences[i].norm(normMean);
      subsequences[i].setLabel(getLabel());
    }

    return subsequences;
  }

  /**
   * The label for supervised data analytics
   * @return
   */
  public String getLabel() {
    return this.label;
  }

  public void setLabel(String label) {
    this.label = label;
  }
}
